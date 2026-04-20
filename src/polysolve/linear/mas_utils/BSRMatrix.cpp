#include <polysolve/linear/mas_utils/BSRMatrix.hpp>

#include <cuda/buffer>
#include <cuda/std/span>
#include <cuda/algorithm>
#include <Eigen/SparseCore>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <polysolve/linear/mas_utils/CudaUtils.cuh>

namespace polysolve::linear::mas
{
    namespace
    {
        template <int BLOCK_DIM>
        struct SmallBlock
        {
            std::array<double, BLOCK_DIM * BLOCK_DIM> data{};
        };

        template <int BLOCK_DIM>
        struct RowBlockEntry
        {
            int col = -1;
            SmallBlock<BLOCK_DIM> block;
        };

        template <int BLOCK_DIM>
        double frobenius_norm(SmallBlock<BLOCK_DIM> const &block)
        {
            double squared_sum = 0.0;
            for (double v : block.data)
            {
                squared_sum += v * v;
            }
            return std::sqrt(squared_sum);
        }

        template <int BLOCK_DIM>
        void append_block_values(SmallBlock<BLOCK_DIM> const &block, std::vector<double> &vals)
        {
            vals.insert(vals.end(), block.data.begin(), block.data.end());
        }

        template <int BLOCK_DIM>
        void build_host_bsr(const StiffnessMatrix &A,
                            ctd::span<const int> permutation,
                            int dim,
                            int &non_zeros,
                            int &topology_non_zeros,
                            std::vector<int> &h_rows,
                            std::vector<int> &h_cols,
                            std::vector<double> &h_vals,
                            std::vector<int> &h_topo_rows,
                            std::vector<int> &h_topo_cols,
                            std::vector<int> &h_topo_weights)
        {
            using RowMajorMatrix =
                Eigen::SparseMatrix<double, Eigen::RowMajor, typename StiffnessMatrix::StorageIndex>;
            using Iter = RowMajorMatrix::InnerIterator;

            RowMajorMatrix Arow = A;
            Arow.makeCompressed();

            std::vector<int> inverse_permutation;
            if (!permutation.empty())
            {
                inverse_permutation.resize(dim);
                for (int old_bi = 0; old_bi < dim; ++old_bi)
                {
                    inverse_permutation[permutation[old_bi]] = old_bi;
                }
            }

            int block_size = BLOCK_DIM * BLOCK_DIM;
            int estimated_blocks = (A.nonZeros() + block_size - 1) / block_size + 1;
            int padded = BLOCK_DIM * dim - A.rows();
            int old_tail_block = dim - 1;

            h_rows.assign(dim + 1, 0);
            h_cols.clear();
            h_cols.reserve(estimated_blocks);
            h_vals.clear();
            h_vals.reserve(estimated_blocks * block_size);
            h_topo_rows.assign(dim + 1, 0);
            h_topo_cols.clear();
            h_topo_cols.reserve(estimated_blocks);
            h_topo_weights.clear();

            std::vector<double> h_topo_norms;
            h_topo_norms.reserve(estimated_blocks);

            for (int bi = 0; bi < dim; ++bi)
            {
                int old_bi = permutation.empty() ? bi : inverse_permutation[bi];
                int scalar_nnz = 0;
                for (int li = 0; li < BLOCK_DIM; ++li)
                {
                    int scalar_row = old_bi * BLOCK_DIM + li;
                    if (scalar_row >= Arow.rows())
                    {
                        break;
                    }
                    scalar_nnz += Arow.outerIndexPtr()[scalar_row + 1] - Arow.outerIndexPtr()[scalar_row];
                }

                std::unordered_map<int, int> col2entry;
                col2entry.reserve((scalar_nnz + BLOCK_DIM - 1) / BLOCK_DIM + 1);
                std::vector<RowBlockEntry<BLOCK_DIM>> row_entries;
                row_entries.reserve((scalar_nnz + BLOCK_DIM - 1) / BLOCK_DIM + 1);

                for (int li = 0; li < BLOCK_DIM; ++li)
                {
                    int scalar_row = old_bi * BLOCK_DIM + li;
                    if (scalar_row >= Arow.rows())
                    {
                        break;
                    }

                    for (Iter it(Arow, scalar_row); it; ++it)
                    {
                        int old_bj = it.col() / BLOCK_DIM;
                        int bj = permutation.empty() ? old_bj : permutation[old_bj];
                        int lj = it.col() - BLOCK_DIM * old_bj;

                        auto map_it = col2entry.find(bj);
                        if (map_it == col2entry.end())
                        {
                            int entry_id = row_entries.size();
                            col2entry.try_emplace(bj, entry_id);
                            row_entries.emplace_back();
                            row_entries.back().col = bj;
                            map_it = col2entry.find(bj);
                        }

                        row_entries[map_it->second].block.data[li * BLOCK_DIM + lj] = it.value();
                    }
                }

                if (padded > 0 && old_bi == old_tail_block)
                {
                    auto map_it = col2entry.find(bi);
                    if (map_it == col2entry.end())
                    {
                        int entry_id = row_entries.size();
                        col2entry.try_emplace(bi, entry_id);
                        row_entries.emplace_back();
                        row_entries.back().col = bi;
                        map_it = col2entry.find(bi);
                    }

                    for (int i = BLOCK_DIM - padded; i < BLOCK_DIM; ++i)
                    {
                        row_entries[map_it->second].block.data[i * BLOCK_DIM + i] = 1.0;
                    }
                }

                std::sort(row_entries.begin(),
                          row_entries.end(),
                          [](RowBlockEntry<BLOCK_DIM> const &a, RowBlockEntry<BLOCK_DIM> const &b) {
                              return a.col < b.col;
                          });

                h_rows[bi + 1] = row_entries.size();
                for (auto const &entry : row_entries)
                {
                    h_cols.push_back(entry.col);
                    append_block_values(entry.block, h_vals);

                    if (bi != entry.col)
                    {
                        h_topo_rows[bi + 1] += 1;
                        h_topo_cols.push_back(entry.col);
                        h_topo_norms.push_back(frobenius_norm(entry.block));
                    }
                }
            }

            for (int i = 0; i < dim; ++i)
            {
                h_rows[i + 1] += h_rows[i];
                h_topo_rows[i + 1] += h_topo_rows[i];
            }

            double topo_strength_sum = 0.0;
            for (double norm : h_topo_norms)
            {
                topo_strength_sum += norm;
            }
            double topo_strength_avg = h_topo_norms.empty() ? 0.0 : topo_strength_sum / h_topo_norms.size();

            h_topo_weights.resize(h_topo_norms.size(), 1);
            for (int i = 0; i < h_topo_norms.size(); ++i)
            {
                int weight = 1;
                if (topo_strength_avg > 0.0)
                {
                    double scaled = h_topo_norms[i] / topo_strength_avg * 100.0;
                    weight = std::clamp(int(std::lround(scaled)), 1, 1000000);
                }
                h_topo_weights[i] = weight;
            }

            non_zeros = h_cols.size();
            topology_non_zeros = h_topo_cols.size();
        }
    } // namespace

    BSRMatrix::BSRMatrix(
        const StiffnessMatrix &A,
        int block_dim,
        ctd::span<const int> permutation,
        CudaRuntime rt)
    {
        if (A.cols() != A.rows() || A.cols() == 0 || A.rows() == 0 || A.nonZeros() == 0)
        {
            throw std::runtime_error("[CudaPcg] Factorization failed due to invalid A");
        }
        if (A.cols() > std::numeric_limits<int>::max() || A.rows() > std::numeric_limits<int>::max())
        {
            throw std::runtime_error("[CudaPcg] A is too large. Row/Col number exceeding int32 max.");
        }

        block_dim_ = block_dim;
        // Pad dimension if neccessary.
        dim_ = (A.rows() + block_dim_ - 1) / block_dim_;
        std::vector<int> h_rows(dim_ + 1, 0);
        std::vector<int> h_cols;
        std::vector<double> h_vals;
        if (block_dim_ == 1)
        {
            build_host_bsr<1>(A,
                              permutation,
                              dim_,
                              non_zeros_,
                              topology_non_zeros_,
                              h_rows,
                              h_cols,
                              h_vals,
                              h_topo_rows_,
                              h_topo_cols_,
                              h_topo_weights_);
        }
        else if (block_dim_ == 2)
        {
            build_host_bsr<2>(A,
                              permutation,
                              dim_,
                              non_zeros_,
                              topology_non_zeros_,
                              h_rows,
                              h_cols,
                              h_vals,
                              h_topo_rows_,
                              h_topo_cols_,
                              h_topo_weights_);
        }
        else if (block_dim_ == 3)
        {
            build_host_bsr<3>(A,
                              permutation,
                              dim_,
                              non_zeros_,
                              topology_non_zeros_,
                              h_rows,
                              h_cols,
                              h_vals,
                              h_topo_rows_,
                              h_topo_cols_,
                              h_topo_weights_);
        }
        else
        {
            throw std::runtime_error("[CudaPcg] Invalid block_dim");
        }

        // Copy host BSR data to device.
        rows_ = cu::make_buffer<int>(rt.stream, rt.mr, h_rows.size(), cu::no_init);
        cu::copy_bytes(rt.stream, h_rows, *rows_);
        cols_ = cu::make_buffer<int>(rt.stream, rt.mr, h_cols.size(), cu::no_init);
        cu::copy_bytes(rt.stream, h_cols, *cols_);
        vals_ = cu::make_buffer<double>(rt.stream, rt.mr, h_vals.size(), cu::no_init);
        cu::copy_bytes(rt.stream, h_vals, *vals_);

        rt.stream.sync();
    }

} // namespace polysolve::linear::mas
