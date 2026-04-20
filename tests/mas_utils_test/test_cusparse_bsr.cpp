#include "TestUtils.hpp"

#include <polysolve/linear/mas_utils/BSRMatrix.hpp>
#include <polysolve/linear/mas_utils/CuSparseWrapper.hpp>

#include <cuda/std/span>

#include <numeric>
#include <vector>

using namespace polysolve;
using namespace polysolve::linear::mas;
using namespace polysolve::linear::mas::test;

namespace
{
    Eigen::MatrixXd build_expected_dense(
        const StiffnessMatrix &A,
        int block_dim,
        ctd::span<const int> permutation)
    {
        int dim = (A.rows() + block_dim - 1) / block_dim;
        int padded_dim = dim * block_dim;
        Eigen::MatrixXd out = Eigen::MatrixXd::Zero(padded_dim, padded_dim);

        for (int k = 0; k < A.outerSize(); ++k)
        {
            for (StiffnessMatrix::InnerIterator it(A, k); it; ++it)
            {
                int old_bi = it.row() / block_dim;
                int old_bj = it.col() / block_dim;
                int bi = permutation.empty() ? old_bi : permutation[old_bi];
                int bj = permutation.empty() ? old_bj : permutation[old_bj];
                int li = it.row() - old_bi * block_dim;
                int lj = it.col() - old_bj * block_dim;
                out(bi * block_dim + li, bj * block_dim + lj) = it.value();
            }
        }

        int padded = padded_dim - A.rows();
        if (padded > 0)
        {
            int old_tail_block = dim - 1;
            int tail_block = permutation.empty() ? old_tail_block : permutation[old_tail_block];
            for (int i = block_dim - padded; i < block_dim; ++i)
            {
                out(tail_block * block_dim + i, tail_block * block_dim + i) = 1.0;
            }
        }

        return out;
    }

    void check_cusparse_spmv(int n, int block_dim, ctd::span<const int> permutation)
    {
        auto rng = make_rng();
        StiffnessMatrix A = make_random_sparse(n, 0.005, rng);

        CudaContext ctx;
        CudaRuntime rt = ctx.rt();

        BSRMatrix M(A, block_dim, permutation, rt);
        BSRView view = M.view();

        int padded_dim = view.dim * block_dim;
        Eigen::VectorXd x = make_random_vector(padded_dim, rng);
        Eigen::VectorXd expected = build_expected_dense(A, block_dim, permutation) * x;

        auto d_x = cu::make_buffer<double>(rt.stream, rt.mr, padded_dim, cu::no_init);
        auto d_y = cu::make_buffer<double>(rt.stream, rt.mr, padded_dim, cu::no_init);
        cu::copy_bytes(rt.stream, ctd::span<const double>(x.data(), padded_dim), d_x);

        CuSparseHandle handle;
        CuSparseBSR mat(view);
        CuSparseConstVec x_desc(ctd::span<const double>(d_x.data(), d_x.size()));
        CuSparseVec y_desc(ctd::span<double>(d_y.data(), d_y.size()));

        double alpha = 1.0;
        double beta = 0.0;
        size_t workspace_size = 0;

        cusparseSetStream(handle.raw, rt.stream.get());
        cusparseSpMV_bufferSize(
            handle.raw,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            mat.raw,
            x_desc.raw,
            &beta,
            y_desc.raw,
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            &workspace_size);

        auto workspace = cu::make_buffer<char>(
            rt.stream,
            rt.mr,
            workspace_size == 0 ? 1 : workspace_size,
            cu::no_init);

        cusparseSpMV(
            handle.raw,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            mat.raw,
            x_desc.raw,
            &beta,
            y_desc.raw,
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            workspace.data());

        std::vector<double> h_y(padded_dim, 0.0);
        cu::copy_bytes(rt.stream, d_y, h_y);
        rt.stream.sync();

        for (int i = 0; i < padded_dim; ++i)
        {
            REQUIRE(h_y[i] == Approx(expected[i]).margin(1e-10).epsilon(1e-10));
        }
    }
} // namespace

TEST_CASE("mas_utils cuSPARSE BSR SpMV block_dim=1", "[mas_utils][cusparse][bsr]")
{
    check_cusparse_spmv(1000, 1, {});
}

TEST_CASE("mas_utils cuSPARSE BSR SpMV block_dim=2", "[mas_utils][cusparse][bsr]")
{
    check_cusparse_spmv(1000, 2, {});
}

TEST_CASE("mas_utils cuSPARSE BSR SpMV block_dim=3", "[mas_utils][cusparse][bsr]")
{
    check_cusparse_spmv(1000, 3, {});
}

TEST_CASE("mas_utils cuSPARSE BSR SpMV permutation", "[mas_utils][cusparse][bsr]")
{
    int n = 127;
    int block_dim = 3;
    int dim = (n + block_dim - 1) / block_dim;

    std::vector<int> permutation(dim);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::reverse(permutation.begin(), permutation.end());

    check_cusparse_spmv(n, block_dim, permutation);
}

TEST_CASE("mas_utils cuSPARSE BSR SpMV permutation block_dim=1", "[mas_utils][cusparse][bsr]")
{
    int n = 127;
    int block_dim = 1;
    int dim = (n + block_dim - 1) / block_dim;

    std::vector<int> permutation(dim);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::reverse(permutation.begin(), permutation.end());

    check_cusparse_spmv(n, block_dim, permutation);
}

TEST_CASE("mas_utils cuSPARSE BSR SpMV permutation block_dim=2", "[mas_utils][cusparse][bsr]")
{
    int n = 127;
    int block_dim = 2;
    int dim = (n + block_dim - 1) / block_dim;

    std::vector<int> permutation(dim);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::reverse(permutation.begin(), permutation.end());

    check_cusparse_spmv(n, block_dim, permutation);
}
