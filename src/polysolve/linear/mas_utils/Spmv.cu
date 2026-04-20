#include <polysolve/linear/mas_utils/Spmv.hpp>

#include <polysolve/linear/mas_utils/BCOOMatrix.hpp>
#include <polysolve/linear/mas_utils/CudaUtils.cuh>
#include <polysolve/linear/mas_utils/SimpleLinalg.cuh>
#include <cuda/std/span>
#include <cuda/atomic>
#include <cuda/algorithm>
#include <cub/cub.cuh>
#include <cassert>

namespace polysolve::linear::mas
{

    namespace
    {
        // template <int D>
        // struct MatPlus
        // {
        //     __device__ Mat<double, D, 1> operator()(
        //         const Mat<double, D, 1> &a,
        //         const Mat<double, D, 1> &b) const
        //     {
        //         return vadd(a, b);
        //     }
        // };

        // Sparse matrix vector product kernel for block COO matrix.
        // Compute y = Ax.
        template <int D>
        __global__ void spmv_kernel(
            BSRView A,
            ctd::span<const double> x,
            ctd::span<double> y)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x; // thread id
            int lid = threadIdx.x % 32;                      // lane id
            int wid = threadIdx.x / 32;                      // warp id
            if (tid >= A.non_zeros)
            {
                return;
            }

            // Load block matrix m for current thread.
            int mat_offset = D * D * tid;
            auto m = MatRef<const double, D, D>::row_major(A.vals.data() + mat_offset);

            int x_offset = A.cols[tid] * D;
            auto x_in = MatRef<const double, D, 1>::row_major(x.data() + x_offset);

            // Compute y=Mx.
            auto result = mat_mul(m, x_in);

            // Use HeadSegmentedSum to reduce y.
            // Within each warp, threads with the same row idx is grouped together.
            // Sum all y in each group before writing them to global memory to reduce atomicAdd overhead.

            // This should just be dummy storage?
            using WarpReduce = cub::WarpReduce<Mat<double, D, 1>>;
            __shared__ typename WarpReduce::TempStorage tmp[4];

            bool is_head = (lid == 0 || tid == 0 || A.rows[tid] != A.rows[tid - 1]);
            auto reduced = WarpReduce(tmp[wid]).HeadSegmentedReduce(result, is_head, vadd<Mat<double, D, 1>, Mat<double, D, 1>>);

            if (is_head)
            {
                int y_offset = A.rows[tid] * D;
                for (int i = 0; i < D; ++i)
                {
                    cu::atomic_ref<double> out{y[y_offset + i]};
                    out += reduced(i);
                }
            }
        }

    } // namespace

    void spmv(BSRView A, ctd::span<const double> x, ctd::span<double> y, CudaRuntime rt)
    {
        cu::fill_bytes(rt.stream, y, 0);

        constexpr int BLOCK_SIZE = 128;
        int grid_size = div_round_upper(A.non_zeros, BLOCK_SIZE);

        auto st = rt.stream.get();

        switch (A.block_dim)
        {
        case 1:
            spmv_kernel<1><<<grid_size, BLOCK_SIZE, 0, st>>>(A, x, y);
            break;
        case 2:
            spmv_kernel<2><<<grid_size, BLOCK_SIZE, 0, st>>>(A, x, y);
            break;
        case 3:
            spmv_kernel<3><<<grid_size, BLOCK_SIZE, 0, st>>>(A, x, y);
            break;
        default:
            assert(false && "Unexpected block dim");
        }
    }

} // namespace polysolve::linear::mas
