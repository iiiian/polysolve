#include "TestUtils.hpp"

#include <polysolve/linear/mas_utils/BCOOMatrix.hpp>

#include <cuda/std/span>

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

using namespace polysolve;
using namespace polysolve::linear::mas;
using namespace polysolve::linear::mas::test;

namespace
{
    void check_bcoo_roundtrip(int n, int block_dim)
    {
        auto rng = make_rng();
        StiffnessMatrix A = make_random_sparse(n, 0.005, rng);

        CudaContext ctx;
        CudaRuntime rt = ctx.rt();

        BCOOMatrix M(A, block_dim, rt);
        BCOOView view = M.view();
        TopologyView host_topology = M.host_topology_view();
        TopologyView device_topology = M.device_topology_view();

        int expected_dim = (n + block_dim - 1) / block_dim;
        int padded_dim = expected_dim * block_dim;

        REQUIRE(view.dim == expected_dim);
        REQUIRE(view.block_dim == block_dim);

        // Compute expected non_zeros: number of unique (bi, bj) blocks that
        // have at least one entry in A.
        std::set<std::pair<int, int>> expected_blocks;
        for (int k = 0; k < A.outerSize(); ++k)
        {
            for (StiffnessMatrix::InnerIterator it(A, k); it; ++it)
            {
                expected_blocks.emplace(
                    it.row() / block_dim,
                    it.col() / block_dim);
            }
        }
        REQUIRE(view.non_zeros == int(expected_blocks.size()));

        // Copy device data back to host.
        std::vector<int> h_rows(view.non_zeros);
        std::vector<int> h_cols(view.non_zeros);
        std::vector<int> h_diag_index(view.dim);
        std::vector<double> h_vals(view.non_zeros * block_dim * block_dim);
        std::vector<int> h_topology_row_ptr(host_topology.dim + 1);
        std::vector<int> h_topology_cols(host_topology.non_zeros);
        std::vector<int> h_topology_row_ptr_from_device(device_topology.dim + 1);
        std::vector<int> h_topology_cols_from_device(device_topology.non_zeros);

        cu::copy_bytes(rt.stream, view.rows, h_rows);
        cu::copy_bytes(rt.stream, view.cols, h_cols);
        cu::copy_bytes(rt.stream, view.diag_index, h_diag_index);
        cu::copy_bytes(rt.stream, view.vals, h_vals);
        std::copy(host_topology.row_ptr.begin(), host_topology.row_ptr.end(), h_topology_row_ptr.begin());
        std::copy(host_topology.cols.begin(), host_topology.cols.end(), h_topology_cols.begin());
        cu::copy_bytes(rt.stream, device_topology.row_ptr, h_topology_row_ptr_from_device);
        cu::copy_bytes(rt.stream, device_topology.cols, h_topology_cols_from_device);
        rt.stream.sync();

        // Row-major sort check (strictly increasing composite key).
        for (int idx = 1; idx < view.non_zeros; ++idx)
        {
            int prev = h_rows[idx - 1] * expected_dim + h_cols[idx - 1];
            int cur = h_rows[idx] * expected_dim + h_cols[idx];
            REQUIRE(cur > prev);
        }

        REQUIRE(host_topology.dim == expected_dim);
        REQUIRE(h_topology_row_ptr.front() == 0);
        REQUIRE(h_topology_row_ptr.back() == host_topology.non_zeros);
        REQUIRE(h_topology_row_ptr_from_device == h_topology_row_ptr);
        REQUIRE(h_topology_cols_from_device == h_topology_cols);

        std::vector<int> expected_topology_row_ptr(expected_dim + 1, 0);
        std::vector<int> expected_topology_cols;
        expected_topology_cols.reserve(view.non_zeros);
        for (int idx = 0; idx < view.non_zeros; ++idx)
        {
            int row = h_rows[idx];
            int col = h_cols[idx];
            if (row == col)
            {
                continue;
            }
            expected_topology_cols.push_back(col);
            expected_topology_row_ptr[row + 1]++;
        }
        for (int row = 0; row < expected_dim; ++row)
        {
            expected_topology_row_ptr[row + 1] += expected_topology_row_ptr[row];
        }

        REQUIRE(host_topology.non_zeros == int(expected_topology_cols.size()));
        REQUIRE(h_topology_row_ptr == expected_topology_row_ptr);
        REQUIRE(h_topology_cols == expected_topology_cols);

        for (int row = 0; row < expected_dim; ++row)
        {
            REQUIRE(h_topology_row_ptr[row] <= h_topology_row_ptr[row + 1]);
            for (int idx = h_topology_row_ptr[row] + 1; idx < h_topology_row_ptr[row + 1]; ++idx)
            {
                REQUIRE(h_topology_cols[idx - 1] < h_topology_cols[idx]);
            }
            for (int idx = h_topology_row_ptr[row]; idx < h_topology_row_ptr[row + 1]; ++idx)
            {
                REQUIRE(h_topology_cols[idx] != row);
            }
        }

        // diag_index check.
        for (int bi = 0; bi < expected_dim; ++bi)
        {
            int di = h_diag_index[bi];
            if (di == -1)
            {
                // There should not be any block with (bi, bi) in the list.
                bool found = false;
                for (int idx = 0; idx < view.non_zeros; ++idx)
                {
                    if (h_rows[idx] == bi && h_cols[idx] == bi)
                    {
                        found = true;
                        break;
                    }
                }
                REQUIRE_FALSE(found);
            }
            else
            {
                REQUIRE(di >= 0);
                REQUIRE(di < view.non_zeros);
                REQUIRE(h_rows[di] == bi);
                REQUIRE(h_cols[di] == bi);
            }
        }

        // Reconstruct a padded dense matrix from the BCOO data (row-major blocks).
        int block_size = block_dim * block_dim;
        Eigen::MatrixXd A_rt = Eigen::MatrixXd::Zero(padded_dim, padded_dim);
        for (int idx = 0; idx < view.non_zeros; ++idx)
        {
            int bi = h_rows[idx];
            int bj = h_cols[idx];
            const double *block_ptr = h_vals.data() + idx * block_size;
            for (int li = 0; li < block_dim; ++li)
            {
                for (int lj = 0; lj < block_dim; ++lj)
                {
                    A_rt(bi * block_dim + li, bj * block_dim + lj) = block_ptr[li * block_dim + lj];
                }
            }
        }

        // Compare reconstructed matrix vs original within [0,n) x [0,n).
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                REQUIRE(A_rt(i, j) == A.coeff(i, j));
            }
        }

        // Check padded rows/cols contain identity values only.
        int pad = padded_dim - n;
        if (pad > 0)
        {
            for (int i = n; i < padded_dim; ++i)
            {
                for (int j = 0; j < padded_dim; ++j)
                {
                    double expected = (i == j) ? 1.0 : 0.0;
                    REQUIRE(A_rt(i, j) == expected);
                }
            }
            for (int j = n; j < padded_dim; ++j)
            {
                for (int i = 0; i < padded_dim; ++i)
                {
                    double expected = (i == j) ? 1.0 : 0.0;
                    REQUIRE(A_rt(i, j) == expected);
                }
            }
        }
    }
} // namespace

TEST_CASE("mas_utils BCOOMatrix round-trip block_dim=1", "[mas_utils][bcoo]")
{
    check_bcoo_roundtrip(1000, 1);
}

TEST_CASE("mas_utils BCOOMatrix round-trip block_dim=2", "[mas_utils][bcoo]")
{
    check_bcoo_roundtrip(1000, 2);
}

TEST_CASE("mas_utils BCOOMatrix round-trip block_dim=3", "[mas_utils][bcoo]")
{
    check_bcoo_roundtrip(1000, 3);
}
