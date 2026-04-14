#include "TestUtils.hpp"

#include <polysolve/linear/mas_utils/Inverse.cuh>
#include <polysolve/linear/mas_utils/SimpleLinalg.cuh>

#include <Eigen/Core>
#include <Eigen/LU>

#include <array>

using namespace polysolve::linear::mas;

namespace
{
    template <int D>
    void run_inverse(const std::array<double, D * D> &in_rm,
                     std::array<double, D * D> &out_rm)
    {
        auto m = MatRef<const double, D, D>::row_major(in_rm.data());
        auto o = MatRef<double, D, D>::row_major(out_rm.data());
        inverse<D>(m, o);
    }

    template <int D>
    void expect_identity(const std::array<double, D * D> &out_rm)
    {
        for (int i = 0; i < D; ++i)
        {
            for (int j = 0; j < D; ++j)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                REQUIRE(out_rm[i * D + j] == expected);
            }
        }
    }

    template <int D>
    void expect_matches_eigen(const std::array<double, D * D> &in_rm,
                              const std::array<double, D * D> &out_rm)
    {
        Eigen::Matrix<double, D, D, Eigen::RowMajor> in_mat;
        for (int i = 0; i < D; ++i)
            for (int j = 0; j < D; ++j)
                in_mat(i, j) = in_rm[i * D + j];

        Eigen::Matrix<double, D, D> expected = in_mat.inverse();

        for (int i = 0; i < D; ++i)
        {
            for (int j = 0; j < D; ++j)
            {
                REQUIRE(out_rm[i * D + j] == Approx(expected(i, j)).margin(1e-12).epsilon(1e-10));
            }
        }
    }
} // namespace

TEST_CASE("mas_utils inverse D=1 near-singular -> identity", "[mas_utils][inverse]")
{
    constexpr std::array<double, 1> in = {{1e-30}};
    std::array<double, 1> out{};
    run_inverse<1>(in, out);
    expect_identity<1>(out);
}

TEST_CASE("mas_utils inverse D=1 typical", "[mas_utils][inverse]")
{
    constexpr std::array<double, 1> in = {{2.5}};
    std::array<double, 1> out{};
    run_inverse<1>(in, out);
    expect_matches_eigen<1>(in, out);
}

TEST_CASE("mas_utils inverse D=2 near-singular -> identity", "[mas_utils][inverse]")
{
    // Rank-1: rows are proportional -> det == 0.
    constexpr std::array<double, 4> in = {{1e-15, 2e-15,
                                           2e-15, 4e-15}};
    std::array<double, 4> out{};
    run_inverse<2>(in, out);
    expect_identity<2>(out);
}

TEST_CASE("mas_utils inverse D=2 typical", "[mas_utils][inverse]")
{
    constexpr std::array<double, 4> in = {{4.0, 7.0,
                                           2.0, 6.0}};
    std::array<double, 4> out{};
    run_inverse<2>(in, out);
    expect_matches_eigen<2>(in, out);
}

TEST_CASE("mas_utils inverse D=3 near-singular -> identity", "[mas_utils][inverse]")
{
    // Rank deficient: rows 2, 3 are scalar multiples of row 1.
    constexpr std::array<double, 9> in = {{1.0, 2.0, 3.0,
                                           2.0, 4.0, 6.0,
                                           3.0, 6.0, 9.0}};
    std::array<double, 9> out{};
    run_inverse<3>(in, out);
    expect_identity<3>(out);
}

TEST_CASE("mas_utils inverse D=3 typical", "[mas_utils][inverse]")
{
    constexpr std::array<double, 9> in = {{1.0, 2.0, 3.0,
                                           0.0, 1.0, 4.0,
                                           5.0, 6.0, 0.0}};
    std::array<double, 9> out{};
    run_inverse<3>(in, out);
    expect_matches_eigen<3>(in, out);
}
