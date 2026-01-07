#ifdef POLYSOLVE_WITH_CUDA_PCG

// nvcc will generate intermedia gcc #line pragma.
// Disable -Wpedantic to avoid vomitting warnings.
// This is a work around for AMGCL secretly enabling -Wpedantic.
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

#include "PCGSolver.hpp"

#include <Eigen/Core>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
// #include <iostream>

using RMatrix3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
using Index = polysolve::StiffnessMatrix::StorageIndex;

// Hashable and comparable key for 2D int32_t matrix index (i, j).
// Pack two index into one uint64_t where the first 32 bits is i and the last 32bit is j.
// Can be easily sorted in row major order.
class Int32IndexKey
{
public:
    uint64_t val = 0;

    Int32IndexKey() = default;
    Int32IndexKey(int32_t i, int32_t j)
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

    friend bool operator==(const Int32IndexKey &a,
                           const Int32IndexKey &b) noexcept
    {
        return a.val == b.val;
    }

    friend bool operator<(const Int32IndexKey &a,
                          const Int32IndexKey &b) noexcept
    {
        return a.val < b.val;
    }
};

namespace std
{
    template <>
    struct hash<Int32IndexKey>
    {
        size_t operator()(Int32IndexKey index) const noexcept
        {
            return hash<uint64_t>{}(index.val);
        }
    };
} // namespace std

// Hashable and comparable key for 2D int64_t matrix index (i, j).
// Can be easily sorted in row major order.
class Int64IndexKey
{
public:
    int64_t i = 0;
    int64_t j = 0;

    Int64IndexKey() = default;
    Int64IndexKey(int64_t i, int64_t j)
    {
        this->i = i;
        this->j = j;
    }

    std::pair<int64_t, int64_t> get_index() const
    {
        return std::make_pair(i, j);
    }

    friend bool operator==(const Int64IndexKey &a,
                           const Int64IndexKey &b) noexcept
    {
        return a.i == b.i && a.j == b.j;
    }

    friend bool operator<(const Int64IndexKey &a,
                          const Int64IndexKey &b) noexcept
    {
        if (a.i != b.i)
        {
            return a.i < b.i;
        }
        return a.j < b.j;
    }
};

namespace std
{
    template <>
    struct hash<Int64IndexKey>
    {
        size_t operator()(Int64IndexKey index) const noexcept
        {
            size_t h1 = hash<int64_t>{}(index.i);
            size_t h2 = hash<int64_t>{}(index.j);

            std::size_t seed = h1;
            seed ^= h2 + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
} // namespace std

namespace polysolve::linear
{

// Check cuda API return. Throw runtime error if result is not cudaSuccess.
#define CHECK_CUDA(val) check_cuda((val), #val, __FILE__, __LINE__)
    void check_cuda(cudaError_t result, char const *const func,
                    const char *const file, int const line)
    {
        if (result != cudaSuccess)
        {
            std::string msg;
            msg += "[CudaPcg] CUDA Error at ";
            msg += file;
            msg += ":";
            msg += std::to_string(line);
            msg += " code=";
            msg += std::to_string(static_cast<unsigned int>(result));
            msg += " \"";
            msg += func;
            msg += "\" : ";
            msg += cudaGetErrorString(result);

            throw std::runtime_error(msg);
        }
    }

    // Get raw device ptr from thrust::device_vector.
    template <typename T>
    T *get_raw(thrust::device_vector<T> &vec)
    {
        return thrust::raw_pointer_cast(vec.data());
    }

    // Get raw device ptr from thrust::device_vector.
    template <typename T>
    const T *get_raw(const thrust::device_vector<T> &vec)
    {
        return thrust::raw_pointer_cast(vec.data());
    }

    // 3x3 block COO sparse matrix.
    class BlockCOOMatrix
    {
    public:
        BlockCOOMatrix() = default;
        POLYSOLVE_DELETE_MOVE_COPY(BlockCOOMatrix)

        Index dim = 0;
        Index non_zeros = 0;
        thrust::device_vector<Index> rows;
        thrust::device_vector<Index> cols;
        // Stores row major 3x3 matrix.
        thrust::device_vector<double> vals;
        // A vector of lenth dim that stores the offset of diagonal block.
        // -1 means diagonal block is zero.
        thrust::device_vector<Index> diag_index;

        // Initialize matrix from host CSC matrix A.
        void init(const StiffnessMatrix &A)
        {
            if (A.cols() != A.rows() || A.cols() == 0 || A.rows() == 0 || A.nonZeros() == 0)
            {
                throw std::runtime_error("[CudaPcg] Factorization failed due to invalid A");
            }

            // Pad dimension if neccessary.
            dim = (A.rows() + 2) / 3;

            using IndexKey = std::conditional_t<std::is_same_v<Index, int32_t>, Int32IndexKey, Int64IndexKey>;
            std::unordered_map<IndexKey, RMatrix3d> block_map;

            // Accumulated non-zero 3x3 blocks.
            using Iter = StiffnessMatrix::InnerIterator;
            for (Index k = 0; k < A.outerSize(); ++k)
            {
                for (Iter it(A, k); it; ++it)
                {
                    // Block index (bi, bj).
                    Index bi = it.row() / 3;
                    Index bj = it.col() / 3;
                    // Block local index (li, lj).
                    Index li = it.row() - 3 * bi;
                    Index lj = it.col() - 3 * bj;

                    auto key = IndexKey(bi, bj);
                    auto block_iter = block_map.try_emplace(key, RMatrix3d::Zero()).first;
                    block_iter->second(li, lj) = it.value();
                }
            }

            // Sort blocks in row major order.
            using BlockTriplet = std::pair<IndexKey, RMatrix3d>;
            std::vector<BlockTriplet> h_blocks;
            for (auto &[k, v] : block_map)
            {
                h_blocks.emplace_back(k, v);
            }
            std::sort(h_blocks.begin(), h_blocks.end(), [](const BlockTriplet &a, const BlockTriplet &b) {
                return a.first < b.first;
            });

            // Prepare host Block COO matrix.
            non_zeros = h_blocks.size();
            std::vector<Index> h_rows(non_zeros);
            std::vector<Index> h_cols(non_zeros);
            std::vector<double> h_vals(9 * non_zeros);
            std::vector<Index> h_diag_index(dim, -1);
            for (Index idx = 0; idx < non_zeros; ++idx)
            {
                auto &[key, mat] = h_blocks[idx];
                auto [bi, bj] = key.get_index();
                h_rows[idx] = bi;
                h_cols[idx] = bj;

                if (bi == bj)
                {
                    h_diag_index[bi] = idx;
                }

                memcpy(h_vals.data() + 9 * idx, mat.data(), 9 * sizeof(double));
            }

            // Pad trailing block.
            if (h_rows[non_zeros - 1] == (dim - 1) && h_cols[non_zeros - 1] == (dim - 1))
            {
                int padded = 3 * dim - static_cast<Index>(A.rows());
                if (padded > 0)
                {
                    double *start = h_vals.data() + 9 * (non_zeros - 1);
                    for (int i = 3 - padded; i < 3; ++i)
                    {
                        int diag_offset = 3 * i + i;
                        start[diag_offset] = 1.0;
                    }
                }
            }

            // Copy host COO data to device.
            rows.resize(non_zeros);
            thrust::copy(h_rows.begin(), h_rows.end(), rows.begin());
            cols.resize(non_zeros);
            thrust::copy(h_cols.begin(), h_cols.end(), cols.begin());
            vals.resize(9 * non_zeros);
            thrust::copy(h_vals.begin(), h_vals.end(), vals.begin());
            diag_index.resize(dim);
            thrust::copy(h_diag_index.begin(), h_diag_index.end(), diag_index.begin());
        }
    };

    // Compute the inverse of 3x3 row major matrix.
    __device__ void invert_3x3(const double *m, double *out)
    {
        double a00 = m[0];
        double a01 = m[1];
        double a02 = m[2];
        double a10 = m[3];
        double a11 = m[4];
        double a12 = m[5];
        double a20 = m[6];
        double a21 = m[7];
        double a22 = m[8];

        double c00 = a11 * a22 - a12 * a21;
        double c01 = a02 * a21 - a01 * a22;
        double c02 = a01 * a12 - a02 * a11;
        double c10 = a12 * a20 - a10 * a22;
        double c11 = a00 * a22 - a02 * a20;
        double c12 = a02 * a10 - a00 * a12;
        double c20 = a10 * a21 - a11 * a20;
        double c21 = a01 * a20 - a00 * a21;
        double c22 = a00 * a11 - a01 * a10;

        double det = a00 * c00 + a01 * c10 + a02 * c20;
        // TODO: use better method to detect small determinant.
        if (std::abs(det) < 1e-20)
        {
            // Since this function is used for computing block diagonal preconditioner only,
            // set result to identity if matrix is almost singular.
            out[0] = 1.0;
            out[1] = 0.0;
            out[2] = 0.0;
            out[3] = 0.0;
            out[4] = 1.0;
            out[5] = 0.0;
            out[6] = 0.0;
            out[7] = 0.0;
            out[8] = 1.0;
        }
        else
        {
            double inv_det = 1.0 / det;
            out[0] = c00 * inv_det;
            out[1] = c01 * inv_det;
            out[2] = c02 * inv_det;
            out[3] = c10 * inv_det;
            out[4] = c11 * inv_det;
            out[5] = c12 * inv_det;
            out[6] = c20 * inv_det;
            out[7] = c21 * inv_det;
            out[8] = c22 * inv_det;
        }
    }

    // Sparse matrix vector product kernel for 3x3 block COO matrix.
    // Compute y = Ax.
    template <int block_size>
    __global__ void spmv_kernel(
        Index non_zeros,
        const Index *rows,
        const Index *cols,
        const double *vals,
        const double *x,
        double *y)
    {
        using BlockLoad = cub::BlockLoad<double, block_size, 9, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        using WarpReduce = cub::WarpReduce<double>;

        __shared__ union
        {
            typename BlockLoad::TempStorage load;
            typename WarpReduce::TempStorage reduce[block_size / 32];
        } temp_storage;

        // Load 3x3 matrix m for current thread.
        // Use cub::BlockLoad to coalesce memory read.
        int mat_offset = blockIdx.x * block_size;
        int block_load_size = 0;
        if (mat_offset < non_zeros)
        {
            int remaining_mats = non_zeros - mat_offset;
            block_load_size = (remaining_mats > block_size ? block_size : remaining_mats) * 9;
        }
        double m[9];
        BlockLoad(temp_storage.load).Load(vals + mat_offset * 9, m, block_load_size);

        __syncthreads();

        int tid = mat_offset + threadIdx.x;
        int lane_id = threadIdx.x % 32;
        int warp_id = threadIdx.x / 32;

        bool is_tid_valid = tid < non_zeros;

        double y0 = 0.0;
        double y1 = 0.0;
        double y2 = 0.0;
        if (is_tid_valid)
        {
            int i = rows[tid];
            int j = cols[tid];

            int x_offset = j * 3;
            double x0 = x[x_offset + 0];
            double x1 = x[x_offset + 1];
            double x2 = x[x_offset + 2];

            // Compute y=Mx.
            y0 = m[0] * x0 + m[1] * x1 + m[2] * x2;
            y1 = m[3] * x0 + m[4] * x1 + m[5] * x2;
            y2 = m[6] * x0 + m[7] * x1 + m[8] * x2;
        }

        // Use HeadSegmentedSum to reduce y.
        // Within each warp, threads with the same row idx is grouped together.
        // Sum all y in each group before writing them to global memory to reduce atomicAdd overhead.

        // If is_head is true this means current thread is the start of a new group.
        bool is_head = (lane_id == 0 || tid == 0 || (is_tid_valid && rows[tid] != rows[tid - 1]));
        double y0_sum = WarpReduce(temp_storage.reduce[warp_id]).HeadSegmentedSum(y0, is_head);
        double y1_sum = WarpReduce(temp_storage.reduce[warp_id]).HeadSegmentedSum(y1, is_head);
        double y2_sum = WarpReduce(temp_storage.reduce[warp_id]).HeadSegmentedSum(y2, is_head);

        if (is_head && is_tid_valid)
        {
            int y_offset = rows[tid] * 3;
            atomicAdd(y + y_offset, y0_sum);
            atomicAdd(y + y_offset + 1, y1_sum);
            atomicAdd(y + y_offset + 2, y2_sum);
        }
    }

    // Compute out = num / denom. Output zero if denom is almost zero.
    //
    // This function is used to compute alpha/beta for pcg solver.
    // When the exact solution is reached because matrix A is trivial (ie. block diagonal),
    // the denom will be roughly zero and the correct solution is to zero alpha/beta
    // so the current solution remain unchanged. We need this becuase we are not doing convergence
    // check every pcg iteration for performance reason.
    cudaError_t scalar_division(const thrust::device_vector<double> &num, const thrust::device_vector<double> &denom, thrust::device_vector<double> &out)
    {
        const double *d_num = get_raw(num);
        const double *d_denom = get_raw(denom);
        double *d_out = get_raw(out);

        auto op = [d_num, d_denom, d_out] __device__(Index idx) {
            double denom = *d_denom;
            // TODO: use better method to detect small denominator.
            if (fabs(denom) < 1e-20)
            {
                *d_out = 0.0;
            }
            else
            {
                *d_out = (*d_num) / denom;
            }
        };

        return cub::DeviceFor::Bulk(1, cuda::proclaim_copyable_arguments(op));
    }

    // Compute block diagonal inverse.
    cudaError_t compute_diag_inv(
        const thrust::device_vector<Index> &diag_index,
        const thrust::device_vector<double> &vals,
        thrust::device_vector<double> &out)
    {
        const Index *d_diag_index = get_raw(diag_index);
        const double *d_vals = get_raw(vals);
        double *d_out = get_raw(out);

        auto op = [d_diag_index, d_vals, d_out] __device__(Index row) {
            Index idx = d_diag_index[row];
            double inv_block[9];

            if (idx == -1)
            {
                inv_block[0] = 1.0;
                inv_block[1] = 0.0;
                inv_block[2] = 0.0;
                inv_block[3] = 0.0;
                inv_block[4] = 1.0;
                inv_block[5] = 0.0;
                inv_block[6] = 0.0;
                inv_block[7] = 0.0;
                inv_block[8] = 1.0;
            }
            else
            {
                const double *block = d_vals + idx * 9;
                invert_3x3(block, inv_block);
            }

            double *dst = d_out + row * 9;
#pragma unroll
            for (int i = 0; i < 9; ++i)
            {
                dst[i] = inv_block[i];
            }
        };

        return cub::DeviceFor::Bulk(diag_index.size(), cuda::proclaim_copyable_arguments(op));
    }

    // Compute y = Ax.
    cudaError_t spmv(
        const BlockCOOMatrix &A, const thrust::device_vector<double> &x, thrust::device_vector<double> &y)
    {
        thrust::fill(y.begin(), y.end(), 0.0);

        Index non_zeros = A.non_zeros;
        const Index *d_rows = get_raw(A.rows);
        const Index *d_cols = get_raw(A.cols);
        const double *d_vals = get_raw(A.vals);
        const double *d_x = get_raw(x);
        double *d_y = get_raw(y);

        constexpr int BLOCK_SIZE = 128;
        int grid_size = (non_zeros + BLOCK_SIZE - 1) / BLOCK_SIZE;
        spmv_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(non_zeros, d_rows, d_cols, d_vals, d_x, d_y);
        return cudaSuccess;
    }

    // Apply preconditioner. Compute y = M^-1 x.
    cudaError_t apply_precond(
        const thrust::device_vector<double> &diag_inv,
        const thrust::device_vector<double> &x,
        thrust::device_vector<double> &y)
    {
        const double *d_diag_inv = get_raw(diag_inv);
        const double *d_x = get_raw(x);
        double *d_y = get_raw(y);

        auto op = [d_diag_inv, d_x, d_y] __device__(Index idx) {
            const double *M = d_diag_inv + idx * 9;
            int offset = idx * 3;

            double r0 = d_x[offset + 0];
            double r1 = d_x[offset + 1];
            double r2 = d_x[offset + 2];

            double z0 = M[0] * r0 + M[1] * r1 + M[2] * r2;
            double z1 = M[3] * r0 + M[4] * r1 + M[5] * r2;
            double z2 = M[6] * r0 + M[7] * r1 + M[8] * r2;

            d_y[offset + 0] = z0;
            d_y[offset + 1] = z1;
            d_y[offset + 2] = z2;
        };

        return cub::DeviceFor::Bulk(x.size() / 3, cuda::proclaim_copyable_arguments(op));
    }

    // Compute y = alpha * x + beta * y.
    // If d_alpha != nullptr, alpha = h_alpha * d_alpha. Else alpha = h_alpha.
    // If d_beta != nullptr, beta = h_beta * d_beta. Else beta = h_beta.
    cudaError_t axpby(
        double h_alpha,
        const double *d_alpha,
        double h_beta,
        const double *d_beta,
        const thrust::device_vector<double> &x,
        thrust::device_vector<double> &y)
    {
        const double *d_x = get_raw(x);
        double *d_y = get_raw(y);

        auto op = [h_alpha, d_alpha, h_beta, d_beta, d_x, d_y] __device__(Index idx) {
            double true_d_alpha = (d_alpha == nullptr) ? 1.0 : *d_alpha;
            double true_d_beta = (d_beta == nullptr) ? 1.0 : *d_beta;
            double alpha = h_alpha * true_d_alpha;
            double beta = h_beta * true_d_beta;
            d_y[idx] = alpha * d_x[idx] + beta * d_y[idx];
        };
        return cub::DeviceFor::Bulk(x.size(), cuda::proclaim_copyable_arguments(op));
    }

    // Compute inner product out = <a, b>.
    cudaError_t inner_product(const thrust::device_vector<double> &a, const thrust::device_vector<double> &b,
                              thrust::device_vector<char> &reduction_storage,
                              thrust::device_vector<double> &out)
    {
        const double *d_a = get_raw(a);
        const double *d_b = get_raw(b);
        double *d_out = get_raw(out);

        auto op = [d_a, d_b] __device__(Index idx) -> double {
            return d_a[idx] * d_b[idx];
        };

        // Counting iterator generate 0, 1, ...
        // Transform iterator apply op on 0, 1, ...
        // So input iterator essentially compute a*b for each element.
        thrust::counting_iterator<Index> counting_iter(0);
        auto input_iter = thrust::make_transform_iterator(counting_iter, op);

        // Resize temp storage size if neccessart.
        size_t temp_size = 0;
        cub::DeviceReduce::Sum(
            nullptr,
            temp_size,
            input_iter,
            d_out,
            a.size());
        if (reduction_storage.size() < temp_size)
        {
            reduction_storage.resize(temp_size);
        }

        void *d_temp = get_raw(reduction_storage);
        size_t storage_bytes = reduction_storage.size();

        return cub::DeviceReduce::Sum(
            d_temp,
            storage_bytes,
            input_iter,
            d_out,
            a.size());
    }

    class CudaPCG::CudaPCGImpl
    {
    public:
        void set_parameters(const json &params)
        {
            if (params.contains("max_iter"))
                max_iter_ = params["max_iter"];
            if (params.contains("relative_tolerance"))
                rel_tol_ = params["relative_tolerance"];
            if (params.contains("absolute_tolerance"))
                abs_tol_ = params["absolute_tolerance"];
            if (params.contains("true_residual_period"))
                true_residual_period_ = params["true_residual_period"];
            if (params.contains("use_preconditioned_residual_norm"))
                use_preconditioned_residual_norm_ = params["use_preconditioned_residual_norm"];
        }

        void get_info(json &params) const
        {
            params["solver_iter"] = iterations_;
            params["solver_error"] = residual_norm_;
            params["solver_status"] = pcg_status_to_string(status_);
        }

        void analyze_pattern(const StiffnessMatrix &, const int) {}

        void factorize(const StiffnessMatrix &A)
        {
            A_.init(A);

            diag_inv_.resize(9 * A_.dim);
            CHECK_CUDA(compute_diag_inv(A_.diag_index, A_.vals, diag_inv_));

            dim_ = A.rows();
            padded_dim_ = 3 * A_.dim;
            x_.resize(padded_dim_);
            b_.resize(padded_dim_);
            r_.resize(padded_dim_);
            p_.resize(padded_dim_);
            z_.resize(padded_dim_);
            Ap_.resize(padded_dim_);

            scalar_rz_.resize(1);
            scalar_pAp_.resize(1);
            scalar_alpha_.resize(1);
            scalar_beta_.resize(1);
            scalar_rz_old_.resize(1);
            scalar_rr_.resize(1);
        }

        void solve(const Eigen::Ref<const Eigen::VectorXd> b,
                   Eigen::Ref<Eigen::VectorXd> x)
        {
            status_ = CudaPCGStatus::Running;

            if (b.size() != x.size() || !check_buffer_size(static_cast<Index>(b.size())))
            {
                throw std::runtime_error("[CudaPCG] Size mismatch. Did you forget to call factorize?");
            }

            double *d_b = get_raw(b_);
            CHECK_CUDA(cudaMemcpy(d_b, b.data(), dim_ * sizeof(double), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemset(d_b + dim_, 0.0, (padded_dim_ - dim_) * sizeof(double)));

            // The solver sometimes fails to converge if we use input x as initial value.
            // Maybe the caller does not initialize x properly?
            // Set initial x to zero to work around this issue for now.

            // TODO: investigate the bug and remove this work around.
            thrust::fill(x_.begin(), x_.end(), 0.0);

            pcg_solve();

            CHECK_CUDA(cudaMemcpy(x.data(), get_raw(x_), dim_ * sizeof(double), cudaMemcpyDeviceToHost));
        }

        void set_tolerance(const double tol)
        {
            rel_tol_ = tol;
        }

    private:
        int max_iter_ = 1e5;
        int true_residual_period_ = 50;
        double abs_tol_ = 1e-20;
        double rel_tol_ = 1e-6;
        bool use_preconditioned_residual_norm_ = true;

        Index dim_ = 0;
        Index padded_dim_ = 0;
        int iterations_ = 0;
        double residual_norm_ = 0.0;
        CudaPCGStatus status_;

        BlockCOOMatrix A_;
        thrust::device_vector<double> diag_inv_;
        thrust::device_vector<double> x_;
        thrust::device_vector<double> b_;
        thrust::device_vector<double> r_;
        thrust::device_vector<double> p_;
        thrust::device_vector<double> z_;
        thrust::device_vector<double> Ap_;

        thrust::device_vector<char> reduction_storage_;
        thrust::device_vector<double> scalar_rz_;
        thrust::device_vector<double> scalar_pAp_;
        thrust::device_vector<double> scalar_alpha_;
        thrust::device_vector<double> scalar_beta_;
        thrust::device_vector<double> scalar_rz_old_;
        thrust::device_vector<double> scalar_rr_;

        bool check_buffer_size(Index n) const
        {
            if (n <= 0)
            {
                return false;
            }
            Index block_n = (n + 2) / 3;
            if (block_n != A_.dim)
            {
                return false;
            }

            if (static_cast<Index>(A_.rows.size()) != A_.non_zeros
                || static_cast<Index>(A_.cols.size()) != A_.non_zeros
                || static_cast<Index>(A_.vals.size()) != 9 * A_.non_zeros
                || static_cast<Index>(A_.diag_index.size()) != A_.dim
                || static_cast<Index>(diag_inv_.size()) != 9 * A_.dim)
            {
                return false;
            }

            Index padded_n = 3 * block_n;
            if (padded_n != static_cast<Index>(x_.size())
                || padded_n != static_cast<Index>(b_.size())
                || padded_n != static_cast<Index>(r_.size())
                || padded_n != static_cast<Index>(p_.size())
                || padded_n != static_cast<Index>(z_.size())
                || padded_n != static_cast<Index>(Ap_.size()))
            {
                return false;
            }

            if (scalar_rz_.size() < 1
                || scalar_pAp_.size() < 1
                || scalar_alpha_.size() < 1
                || scalar_beta_.size() < 1
                || scalar_rz_old_.size() < 1
                || scalar_rr_.size() < 1)
            {
                return false;
            }
            return true;
        }

        void pcg_solve()
        {

            // Compute initial residual r = b-Ax.
            CHECK_CUDA(spmv(A_, x_, r_));
            CHECK_CUDA(axpby(1.0, nullptr, -1.0, nullptr, b_, r_));

            // Compute z = M^-1 r.
            CHECK_CUDA(apply_precond(diag_inv_, r_, z_));
            // Initial search direction p = z;
            thrust::copy(z_.begin(), z_.end(), p_.begin());

            // Compute rz = r^T M^-1 r.
            CHECK_CUDA(inner_product(r_, z_, reduction_storage_, scalar_rz_));
            double rz0 = scalar_rz_[0];
            if (std::isnan(rz0) || !std::isfinite(rz0))
            {
                throw std::runtime_error("[CudaPCG] Invalid initial residual.");
            }

            // Compute rr = r^T r.
            double rr0 = 0.0;
            if (!use_preconditioned_residual_norm_)
            {
                CHECK_CUDA(inner_product(r_, r_, reduction_storage_, scalar_rr_));
                rr0 = scalar_rr_[0];
            }

            for (int k = 1; k <= max_iter_; ++k)
            {
                // Compute Ap = A p.
                CHECK_CUDA(spmv(A_, p_, Ap_));
                // Compute pAp = p^T * A * p.
                CHECK_CUDA(inner_product(p_, Ap_, reduction_storage_, scalar_pAp_));
                // Compute alpha = (r M^-1 r) / (p^T A p).
                CHECK_CUDA(scalar_division(scalar_rz_, scalar_pAp_, scalar_alpha_));
                // Compute x = x + alpha A p.
                CHECK_CUDA(axpby(1.0, get_raw(scalar_alpha_), 1.0, nullptr, p_, x_));

                // Compute residual b-Ax directly.
                if (k != 0 && k % true_residual_period_ == 0)
                {
                    CHECK_CUDA(spmv(A_, x_, r_));
                    CHECK_CUDA(axpby(1.0, nullptr, -1.0, nullptr, b_, r_));
                }
                // Compute residual update using r' = r - alpha A p.
                // This saves one spmv but accumulates floating point error overtime.
                else
                {
                    CHECK_CUDA(axpby(-1.0, get_raw(scalar_alpha_), 1.0, nullptr, Ap_, r_));
                }

                // Compute z = M^-1 r.
                CHECK_CUDA(apply_precond(diag_inv_, r_, z_));
                // Compute rz = r M^-1 r.
                CHECK_CUDA(cudaMemcpy(get_raw(scalar_rz_old_), get_raw(scalar_rz_), sizeof(double), cudaMemcpyDeviceToDevice));
                CHECK_CUDA(inner_product(r_, z_, reduction_storage_, scalar_rz_));

                iterations_ = k;
                bool converged = false;

                // Check convergence every 10 iterations.
                if (k % 10 == 0)
                {
                    if (use_preconditioned_residual_norm_)
                    {
                        double rz_new = scalar_rz_[0];
                        residual_norm_ = std::sqrt(rz_new);
                        if (rz_new <= rel_tol_ * rel_tol_ * rz0 || rz_new <= abs_tol_ * abs_tol_)
                        {
                            status_ = (rz_new <= abs_tol_ * abs_tol_) ? CudaPCGStatus::ReachAbsoluteTolerance : CudaPCGStatus::ReachRelativeTolerance;
                            converged = true;
                        }
                    }
                    else
                    {
                        CHECK_CUDA(inner_product(r_, r_, reduction_storage_, scalar_rr_));
                        double rr = scalar_rr_[0];
                        residual_norm_ = std::sqrt(rr);
                        if (rr <= rel_tol_ * rel_tol_ * rr0 || rr <= abs_tol_ * abs_tol_)
                        {
                            status_ = (rr <= abs_tol_ * abs_tol_) ? CudaPCGStatus::ReachAbsoluteTolerance : CudaPCGStatus::ReachRelativeTolerance;
                            converged = true;
                        }
                    }
                }

                if (converged)
                {
                    break;
                }

                // Compute beta = rz / rz_old.
                CHECK_CUDA(scalar_division(scalar_rz_, scalar_rz_old_, scalar_beta_));
                // Compute direction update p' = M^-1 r + beta p.
                CHECK_CUDA(axpby(1.0, nullptr, 1.0, get_raw(scalar_beta_), z_, p_));
            }

            if (iterations_ == max_iter_)
            {
                status_ = CudaPCGStatus::ReachMaxIterations;
            }

            std::cout << "PCG: iter " << iterations_ << ", err " << residual_norm_ << ", stat " << pcg_status_to_string(status_) << std::endl;
        }
    };

    CudaPCG::CudaPCG()
        : impl_(std::make_unique<CudaPCGImpl>())
    {
    }

    CudaPCG::~CudaPCG() = default;

    void CudaPCG::set_parameters(const json &params)
    {
        const std::string solver_name = name();
        if (!params.contains(solver_name))
        {
            return;
        }

        impl_->set_parameters(params[solver_name]);
    }

    void CudaPCG::get_info(json &params) const
    {
        impl_->get_info(params);
    }

    void CudaPCG::analyze_pattern(const StiffnessMatrix &A, const int precond_num)
    {
        impl_->analyze_pattern(A, precond_num);
    }

    void CudaPCG::factorize(const StiffnessMatrix &A)
    {
        impl_->factorize(A);
    }

    void CudaPCG::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        impl_->solve(b, x);
    }

    void CudaPCG::set_tolerance(const double tol)
    {
        impl_->set_tolerance(tol);
    }

    std::string CudaPCG::name() const
    {
        return "CUDA_PCG";
    }

} // namespace polysolve::linear

#endif
