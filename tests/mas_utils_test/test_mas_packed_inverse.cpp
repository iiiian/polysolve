#include "TestUtils.hpp"

#include <polysolve/linear/mas_utils/MASPreconditionerTest.cuh>

#include <Eigen/Core>
#include <Eigen/LU>

#include <vector>

using namespace polysolve::linear::mas;

namespace
{
    int upper_index(int dim, int i, int j)
    {
        if (i > j)
        {
            int tmp = i;
            i = j;
            j = tmp;
        }
        return i * dim - (i * (i + 1) / 2) + j;
    }

    template <int D>
    Eigen::Matrix<double, D, D, Eigen::RowMajor> make_spd_matrix(std::mt19937 &rng)
    {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        Eigen::Matrix<double, D, D, Eigen::RowMajor> M;
        for (int i = 0; i < D; ++i)
        {
            for (int j = 0; j < D; ++j)
            {
                M(i, j) = dist(rng);
            }
        }

        Eigen::Matrix<double, D, D, Eigen::RowMajor> A = M.transpose() * M;
        A.diagonal().array() += double(D);
        return A;
    }

    template <int D>
    void pack_upper(Eigen::Matrix<double, D, D, Eigen::RowMajor> const &A,
                    ctd::span<double> packed)
    {
        for (int i = 0; i < D; ++i)
        {
            for (int j = i; j < D; ++j)
            {
                packed[upper_index(D, i, j)] = A(i, j);
            }
        }
    }

    template <int D>
    void run_packed_inverse_case()
    {
        auto rng = polysolve::linear::mas::test::make_rng();
        polysolve::linear::mas::test::CudaContext ctx;
        CudaRuntime rt = ctx.rt();

        int mat_num = 2;
        int packed_size = D * (D + 1) / 2;
        std::vector<double> h_packed(mat_num * packed_size, 0.0);
        std::vector<Eigen::Matrix<double, D, D, Eigen::RowMajor>> expected_inv(mat_num);

        for (int mat_id = 0; mat_id < mat_num; ++mat_id)
        {
            auto A = make_spd_matrix<D>(rng);
            expected_inv[mat_id] = A.inverse();
            pack_upper(A, ctd::span<double>(h_packed.data() + mat_id * packed_size, packed_size));
        }

        auto d_packed = cu::make_buffer<double>(rt.stream, rt.mr, h_packed.size(), cu::no_init);
        cu::copy_bytes(rt.stream, h_packed, d_packed);

        invert_packed_matrices_for_test(ctd::span<double>(d_packed.data(), d_packed.size()), D / 32, rt);

        std::vector<double> h_inverse(h_packed.size(), 0.0);
        cu::copy_bytes(rt.stream, d_packed, h_inverse);
        rt.stream.sync();

        for (int mat_id = 0; mat_id < mat_num; ++mat_id)
        {
            INFO("matrix: " + std::to_string(mat_id));
            for (int i = 0; i < D; ++i)
            {
                for (int j = i; j < D; ++j)
                {
                    int idx = mat_id * packed_size + upper_index(D, i, j);
                    REQUIRE(h_inverse[idx] == Approx(expected_inv[mat_id](i, j)).margin(1e-10).epsilon(1e-10));
                }
            }
        }
    }

    template <int D>
    void run_packed_symv_case()
    {
        auto rng = polysolve::linear::mas::test::make_rng();
        polysolve::linear::mas::test::CudaContext ctx;
        CudaRuntime rt = ctx.rt();

        int mat_num = 2;
        int packed_size = D * (D + 1) / 2;
        std::vector<double> h_packed(mat_num * packed_size, 0.0);
        std::vector<double> h_x(mat_num * D, 0.0);
        std::vector<double> h_expected(mat_num * D, 0.0);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int mat_id = 0; mat_id < mat_num; ++mat_id)
        {
            auto A = make_spd_matrix<D>(rng);
            pack_upper(A, ctd::span<double>(h_packed.data() + mat_id * packed_size, packed_size));

            Eigen::Matrix<double, D, 1> x;
            for (int i = 0; i < D; ++i)
            {
                double v = dist(rng);
                x(i) = v;
                h_x[mat_id * D + i] = v;
            }

            Eigen::Matrix<double, D, 1> y = A * x;
            for (int i = 0; i < D; ++i)
            {
                h_expected[mat_id * D + i] = y(i);
            }
        }

        auto d_x = cu::make_buffer<double>(rt.stream, rt.mr, h_x.size(), cu::no_init);
        auto d_y = cu::make_buffer<double>(rt.stream, rt.mr, h_x.size(), cu::no_init);
        cu::copy_bytes(rt.stream, h_x, d_x);
        cu::fill_bytes(rt.stream, d_y, 0.0);

        apply_packed_matrices_for_test(
            ctd::span<const double>(h_packed.data(), h_packed.size()),
            ctd::span<const double>(d_x.data(), d_x.size()),
            ctd::span<double>(d_y.data(), d_y.size()),
            D / 32,
            rt);

        std::vector<double> h_y(h_x.size(), 0.0);
        cu::copy_bytes(rt.stream, d_y, h_y);
        rt.stream.sync();

        for (int i = 0; i < h_y.size(); ++i)
        {
            REQUIRE(h_y[i] == Approx(h_expected[i]).margin(1e-10).epsilon(1e-10));
        }
    }
} // namespace

TEST_CASE("mas_utils packed inverse 32x32 matches Eigen", "[mas_utils][inverse][mas]")
{
    run_packed_inverse_case<32>();
}

TEST_CASE("mas_utils packed inverse 64x64 matches Eigen", "[mas_utils][inverse][mas]")
{
    run_packed_inverse_case<64>();
}

TEST_CASE("mas_utils packed inverse 96x96 matches Eigen", "[mas_utils][inverse][mas]")
{
    run_packed_inverse_case<96>();
}

TEST_CASE("mas_utils packed symv 32x32 matches Eigen", "[mas_utils][symv][mas]")
{
    run_packed_symv_case<32>();
}

TEST_CASE("mas_utils packed symv 64x64 matches Eigen", "[mas_utils][symv][mas]")
{
    run_packed_symv_case<64>();
}

TEST_CASE("mas_utils packed symv 96x96 matches Eigen", "[mas_utils][symv][mas]")
{
    run_packed_symv_case<96>();
}
