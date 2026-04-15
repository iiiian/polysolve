#pragma once

#include <cuda/buffer>
#include <cuda/std/span>

#include <polysolve/linear/mas_utils/CudaUtils.cuh>
#include <polysolve/Types.hpp>

#include <vector>

namespace polysolve::linear::mas
{
    struct BCOOView
    {
        int dim;
        int block_dim;
        int non_zeros;
        ctd::span<const int> rows;
        ctd::span<const int> cols;
        ctd::span<const int> diag_index;
        ctd::span<const double> vals;
    };

    // CSR topology
    struct TopologyView
    {
        int dim;
        int non_zeros;
        ctd::span<const int> row_ptr;
        ctd::span<const int> cols;
    };

    class BCOOMatrix
    {
    public:
        BCOOMatrix() = default;

        /// @brief Build from host CSR matrix. Sync stream before return.
        BCOOMatrix(const StiffnessMatrix &A, int block_dim, CudaRuntime rt);

        BCOOView view() const
        {
            return BCOOView{dim_, block_dim_, non_zeros_, *rows_, *cols_, *diag_index_, *vals_};
        }

        TopologyView host_topology_view() const
        {
            return TopologyView{
                dim_,
                non_zeros_,
                h_csr_row_ptr_,
                h_cols_,
            };
        }

        TopologyView device_topology_view() const
        {
            return TopologyView{
                dim_,
                non_zeros_,
                *csr_row_ptr_,
                *cols_,
            };
        }

    private:
        int dim_ = 0;
        int block_dim_ = 0;
        int non_zeros_ = 0;
        Buf<int> rows_;
        Buf<int> cols_;
        Buf<int> diag_index_;
        Buf<double> vals_;

        Buf<int> csr_row_ptr_;
        std::vector<int> h_csr_row_ptr_;
        std::vector<int> h_cols_;
    };

} // namespace polysolve::linear::mas
