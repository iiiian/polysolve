#pragma once

#include <cstdint>
#include <cuda/std/span>

#include <vector>

#include <polysolve/Types.hpp>
#include <polysolve/linear/mas_utils/BCOOMatrix.hpp>
#include <polysolve/linear/mas_utils/CudaUtils.cuh>

namespace polysolve::linear::mas
{
    class MASPreconditioner
    {
    public:
        void factorize(const StiffnessMatrix &A, BCOOView matrix, TopologyView topology, CudaRuntime rt);

        void apply(ctd::span<const double> r, ctd::span<double> z, CudaRuntime rt);

        bool empty() const
        {
            return !fine_ancestors_ || !level_node_offsets_device_ || !local_matrices_
                   || !inverse_matrices_ || !multi_level_r_ || !multi_level_z_;
        }

        int block_dim() const { return block_dim_; }
        int level_count() const { return level_count_; }

    private:
        struct LevelInfo
        {
            int node_count = 0;
            int padded_count = 0;
            int bank_count = 0;
        };

        int block_dim_ = 0;
        int fine_node_count_ = 0;
        int level_count_ = 0;
        int total_padded_nodes_ = 0;
        int max_bank_count_ = 0;

        std::vector<int> cached_row_ptr_;
        std::vector<int> cached_cols_;
        std::vector<LevelInfo> levels_;
        std::vector<int> level_node_offsets_;
        std::vector<int> level_matrix_offsets_;

        Buf<int> fine_ancestors_;
        Buf<int> level_node_offsets_device_;
        Buf<int> level_matrix_offsets_device_;
        Buf<int> level0_map_;
        Buf<int> level0_remap_;
        Buf<uint32_t> connect_masks_;
        Buf<int> component_counts_;
        Buf<int> bank_offsets_;
        Buf<int> current_to_next_;
        Buf<int> next_count_;
        Buf<char> scan_storage_;
        Buf<double> local_matrices_;
        Buf<double> inverse_matrices_;
        Buf<double> multi_level_r_;
        Buf<double> multi_level_z_;

        void analyze_pattern(
            TopologyView topology,
            const std::vector<int> &row_ptr,
            const std::vector<int> &cols,
            const std::vector<int> &level0_map,
            const std::vector<int> &level0_remap,
            int level0_bank_count,
            CudaRuntime rt);
        bool same_pattern(const std::vector<int> &row_ptr, const std::vector<int> &cols) const;
    };

} // namespace polysolve::linear::mas
