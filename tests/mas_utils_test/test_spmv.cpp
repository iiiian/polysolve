#include "TestUtils.hpp"

#include <polysolve/linear/mas_utils/BCOOMatrix.hpp>
#include <polysolve/linear/mas_utils/Spmv.hpp>

#include <cuda/buffer>
#include <cuda/std/span>

#include <vector>

using namespace polysolve;
using namespace polysolve::linear::mas;
using namespace polysolve::linear::mas::test;

namespace
{
    void check_spmv(int n, int block_dim)
    {
        auto rng = make_rng();
        StiffnessMatrix A = make_random_sparse(n, 0.005, rng);
        Eigen::VectorXd x = make_random_vector(n, rng);

        Eigen::VectorXd y_ref = A * x;

        CudaContext ctx;
        CudaRuntime rt = ctx.rt();

        BSRMatrix M(A, block_dim, rt);
        BSRView view = M.view();
        int padded = view.dim * view.block_dim;

        Eigen::VectorXd x_padded = Eigen::VectorXd::Zero(padded);
        x_padded.head(n) = x;

        auto x_dev = cu::make_buffer<double>(rt.stream, rt.mr, padded, cu::no_init);
        auto y_dev = cu::make_buffer<double>(rt.stream, rt.mr, padded, cu::no_init);

        cu::copy_bytes(rt.stream,
                       ctd::span<const double>(x_padded.data(), padded),
                       x_dev);

        spmv(view, x_dev, y_dev, rt);

        std::vector<double> y_host(padded, 0.0);
        cu::copy_bytes(rt.stream, y_dev, y_host);
        rt.stream.sync();

        for (int i = 0; i < n; ++i)
        {
            REQUIRE(y_host[i] == Approx(y_ref[i]).margin(1e-9).epsilon(1e-10));
        }
    }
} // namespace

TEST_CASE("mas_utils spmv block_dim=1", "[mas_utils][spmv]")
{
    check_spmv(1000, 1);
}

TEST_CASE("mas_utils spmv block_dim=2", "[mas_utils][spmv]")
{
    check_spmv(1000, 2);
}

TEST_CASE("mas_utils spmv block_dim=3", "[mas_utils][spmv]")
{
    check_spmv(1000, 3);
}
