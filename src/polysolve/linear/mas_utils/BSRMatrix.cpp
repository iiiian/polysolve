#include <polysolve/linear/mas_utils/BSRMatrix.hpp>

#include <cuda/buffer>
#include <cuda/std/span>
#include <cuda/algorithm>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <polysolve/linear/mas_utils/CudaUtils.cuh>

// Hashable and comparable key for 2D int32_t matrix index (i, j).
// Pack two index into one uint64_t where the first 32 bits is i and the last 32bit is j.
// Can be easily sorted in row major order.
class IndexKey
{
public:
    uint64_t val = 0;

    IndexKey() = default;
    IndexKey(int32_t i, int32_t j)
    {
        uint64_t a = static_cast<uint64_t>(i) << 32;
        uint64_t b = static_cast<uint64_t>(j);
        val = a | b;
    }

    std::pair<int32_t, int32_t> get_index() const
    {
        int32_t i = static_cast<int32_t>(val >> 32);
        int32_t j = static_cast<int32_t>(val & 0xFFFFFFFFULL);
        return std::make_pair(i, j);
    }

    friend bool operator==(const IndexKey &a,
                           const IndexKey &b) noexcept
    {
        return a.val == b.val;
    }

    friend bool operator<(const IndexKey &a,
                          const IndexKey &b) noexcept
    {
        return a.val < b.val;
    }
};

namespace std
{
    template <>
    struct hash<IndexKey>
    {
        size_t operator()(IndexKey index) const noexcept
        {
            return hash<uint64_t>{}(index.val);
        }
    };
} // namespace std

namespace polysolve::linear::mas
{
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

        using Block = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        std::unordered_map<IndexKey, Block> block_map;

        // Accumulated non-zero blocks.
        using Iter = StiffnessMatrix::InnerIterator;
        for (int k = 0; k < A.outerSize(); ++k)
        {
            for (Iter it(A, k); it; ++it)
            {
                // Block index (bi, bj).
                int old_bi = it.row() / block_dim_;
                int old_bj = it.col() / block_dim_;
                int bi = permutation.empty() ? old_bi : permutation[old_bi];
                int bj = permutation.empty() ? old_bj : permutation[old_bj];
                // Block local index (li, lj).
                int li = it.row() - block_dim_ * old_bi;
                int lj = it.col() - block_dim_ * old_bj;

                auto key = IndexKey(bi, bj);
                auto block_iter = block_map.try_emplace(key, Block::Zero(block_dim_, block_dim_)).first;
                block_iter->second(li, lj) = it.value();
            }
        }

        int padded = block_dim_ * dim_ - A.rows();
        if (padded > 0)
        {
            int old_tail_block = dim_ - 1;
            int tail_block = permutation.empty() ? old_tail_block : permutation[old_tail_block];
            auto key = IndexKey(tail_block, tail_block);
            auto block_iter = block_map.try_emplace(key, Block::Zero(block_dim_, block_dim_)).first;
            for (int i = block_dim_ - padded; i < block_dim_; ++i)
            {
                block_iter->second(i, i) = 1.0;
            }
        }

        // Sort blocks in row major order.
        using BlockTriplet = std::pair<IndexKey, Block>;
        std::vector<BlockTriplet> h_blocks;
        for (auto &[k, v] : block_map)
        {
            h_blocks.emplace_back(k, v);
        }
        std::sort(h_blocks.begin(), h_blocks.end(), [](const BlockTriplet &a, const BlockTriplet &b) {
            return a.first < b.first;
        });

        // Prepare host BSR matrix.
        non_zeros_ = h_blocks.size();
        int block_size = block_dim_ * block_dim_;
        std::vector<int> h_rows(dim_ + 1, 0);
        std::vector<int> h_cols;
        h_cols.reserve(non_zeros_);
        std::vector<double> h_vals(block_size * non_zeros_);
        h_topo_rows_.resize(dim_ + 1, 0);
        double topo_strength_sum = 0.0;
        int topo_edge_num = 0;

        for (auto &[key, mat] : h_blocks)
        {
            auto [bi, bj] = key.get_index();
            if (bi == bj)
            {
                continue;
            }

            topo_strength_sum += mat.norm();
            topo_edge_num += 1;
        }
        double topo_strength_avg = topo_edge_num > 0 ? topo_strength_sum / topo_edge_num : 0.0;

        for (int idx = 0; idx < non_zeros_; ++idx)
        {
            auto &[key, mat] = h_blocks[idx];
            auto [bi, bj] = key.get_index();

            h_rows[bi + 1] += 1;
            h_cols.push_back(bj);

            if (bi != bj)
            {
                h_topo_rows_[bi + 1] += 1;
                h_topo_cols_.push_back(bj);

                int weight = 1;
                if (topo_strength_avg > 0.0)
                {
                    double scaled = mat.norm() / topo_strength_avg * 100.0;
                    weight = std::clamp(int(std::lround(scaled)), 1, 1000000);
                }
                h_topo_weights_.push_back(weight);
            }

            memcpy(h_vals.data() + block_size * idx, mat.data(), block_size * sizeof(double));
        }

        for (int i = 0; i < dim_; ++i)
        {
            h_rows[i + 1] += h_rows[i];
            h_topo_rows_[i + 1] += h_topo_rows_[i];
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
