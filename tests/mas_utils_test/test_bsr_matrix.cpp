#include "TestUtils.hpp"

#include <polysolve/linear/mas_utils/BSRMatrix.hpp>

#include <cuda/std/span>

#include <numeric>
#include <set>
#include <utility>
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

    void check_topology(
        const Eigen::MatrixXd &expected,
        int dim,
        int block_dim,
        TopologyView topology)
    {
        std::vector<int> expected_row_ptr(dim + 1, 0);
        std::vector<int> expected_cols;

        for (int bi = 0; bi < dim; ++bi)
        {
            for (int bj = 0; bj < dim; ++bj)
            {
                if (bi == bj)
                {
                    continue;
                }

                bool non_zero = false;
                for (int li = 0; li < block_dim && !non_zero; ++li)
                {
                    for (int lj = 0; lj < block_dim; ++lj)
                    {
                        if (expected(bi * block_dim + li, bj * block_dim + lj) != 0.0)
                        {
                            non_zero = true;
                            break;
                        }
                    }
                }

                if (non_zero)
                {
                    expected_cols.push_back(bj);
                    expected_row_ptr[bi + 1] += 1;
                }
            }
        }

        for (int bi = 0; bi < dim; ++bi)
        {
            expected_row_ptr[bi + 1] += expected_row_ptr[bi];
        }

        REQUIRE(topology.row_ptr.size() == expected_row_ptr.size());
        REQUIRE(topology.cols.size() == expected_cols.size());
        REQUIRE(std::equal(topology.row_ptr.begin(), topology.row_ptr.end(), expected_row_ptr.begin()));
        REQUIRE(std::equal(topology.cols.begin(), topology.cols.end(), expected_cols.begin()));
    }

    void check_bsr_roundtrip(int n, int block_dim, ctd::span<const int> permutation)
    {
        auto rng = make_rng();
        StiffnessMatrix A = make_random_sparse(n, 0.005, rng);

        CudaContext ctx;
        CudaRuntime rt = ctx.rt();

        BSRMatrix M(A, block_dim, permutation, rt);
        BSRView view = M.view();
        TopologyView topology = M.host_topology_view();

        int dim = (n + block_dim - 1) / block_dim;
        int padded_dim = dim * block_dim;
        REQUIRE(view.dim == dim);
        REQUIRE(view.block_dim == block_dim);
        REQUIRE(view.rows.size() == dim + 1);

        std::vector<int> h_rows(dim + 1);
        std::vector<int> h_cols(view.non_zeros);
        std::vector<double> h_vals(view.non_zeros * block_dim * block_dim);
        cu::copy_bytes(rt.stream, view.rows, h_rows);
        cu::copy_bytes(rt.stream, view.cols, h_cols);
        cu::copy_bytes(rt.stream, view.vals, h_vals);
        rt.stream.sync();

        REQUIRE(h_rows.front() == 0);
        REQUIRE(h_rows.back() == view.non_zeros);
        for (int bi = 0; bi < dim; ++bi)
        {
            REQUIRE(h_rows[bi] <= h_rows[bi + 1]);
            for (int idx = h_rows[bi] + 1; idx < h_rows[bi + 1]; ++idx)
            {
                REQUIRE(h_cols[idx - 1] < h_cols[idx]);
            }
        }

        Eigen::MatrixXd actual = Eigen::MatrixXd::Zero(padded_dim, padded_dim);
        int block_size = block_dim * block_dim;
        for (int bi = 0; bi < dim; ++bi)
        {
            for (int idx = h_rows[bi]; idx < h_rows[bi + 1]; ++idx)
            {
                int bj = h_cols[idx];
                const double *block = h_vals.data() + idx * block_size;
                for (int li = 0; li < block_dim; ++li)
                {
                    for (int lj = 0; lj < block_dim; ++lj)
                    {
                        actual(bi * block_dim + li, bj * block_dim + lj) =
                            block[li * block_dim + lj];
                    }
                }
            }
        }

        Eigen::MatrixXd expected = build_expected_dense(A, block_dim, permutation);
        REQUIRE(actual.rows() == expected.rows());
        REQUIRE(actual.cols() == expected.cols());
        for (int i = 0; i < padded_dim; ++i)
        {
            for (int j = 0; j < padded_dim; ++j)
            {
                REQUIRE(actual(i, j) == Approx(expected(i, j)).margin(1e-12));
            }
        }

        std::set<std::pair<int, int>> expected_blocks;
        for (int bi = 0; bi < dim; ++bi)
        {
            for (int bj = 0; bj < dim; ++bj)
            {
                bool non_zero = false;
                for (int li = 0; li < block_dim && !non_zero; ++li)
                {
                    for (int lj = 0; lj < block_dim; ++lj)
                    {
                        if (expected(bi * block_dim + li, bj * block_dim + lj) != 0.0)
                        {
                            non_zero = true;
                            break;
                        }
                    }
                }

                if (non_zero)
                {
                    expected_blocks.emplace(bi, bj);
                }
            }
        }
        REQUIRE(view.non_zeros == expected_blocks.size());

        check_topology(expected, dim, block_dim, topology);
    }
} // namespace

TEST_CASE("mas_utils BSRMatrix round-trip block_dim=1", "[mas_utils][bsr]")
{
    check_bsr_roundtrip(1000, 1, {});
}

TEST_CASE("mas_utils BSRMatrix round-trip block_dim=2", "[mas_utils][bsr]")
{
    check_bsr_roundtrip(1000, 2, {});
}

TEST_CASE("mas_utils BSRMatrix round-trip block_dim=3", "[mas_utils][bsr]")
{
    check_bsr_roundtrip(1000, 3, {});
}

TEST_CASE("mas_utils BSRMatrix permutation", "[mas_utils][bsr]")
{
    int n = 127;
    int block_dim = 3;
    int dim = (n + block_dim - 1) / block_dim;

    std::vector<int> permutation(dim);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::reverse(permutation.begin(), permutation.end());

    check_bsr_roundtrip(n, block_dim, permutation);
}
