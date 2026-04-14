#include "TestUtils.hpp"

#include <polysolve/linear/mas_utils/InnerProduct.hpp>

#include <cuda/buffer>
#include <cuda/std/span>

#include <vector>

using namespace polysolve::linear::mas;
using namespace polysolve::linear::mas::test;

TEST_CASE("mas_utils inner_product matches Eigen", "[mas_utils][inner_product]")
{
    constexpr int n = 1000;

    auto rng = make_rng();
    Eigen::VectorXd a = make_random_vector(n, rng);
    Eigen::VectorXd b = make_random_vector(n, rng);

    CudaContext ctx;
    CudaRuntime rt = ctx.rt();

    auto a_dev = cu::make_buffer<double>(rt.stream, rt.mr, n, cu::no_init);
    auto b_dev = cu::make_buffer<double>(rt.stream, rt.mr, n, cu::no_init);
    auto out_dev = cu::make_buffer<double>(rt.stream, rt.mr, 1, cu::no_init);

    cu::copy_bytes(rt.stream, ctd::span<const double>(a.data(), n), a_dev);
    cu::copy_bytes(rt.stream, ctd::span<const double>(b.data(), n), b_dev);

    inner_product(a_dev, b_dev, out_dev, rt);

    double out_host = 0.0;
    cu::copy_bytes(rt.stream, out_dev, ctd::span<double>(&out_host, 1));
    rt.stream.sync();

    double expected = a.dot(b);
    REQUIRE(out_host == Approx(expected).margin(1e-9).epsilon(1e-12));
}
