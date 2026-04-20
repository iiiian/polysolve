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
        void analyze_pattern(
            const BSRMatrix &A, CudaRuntime rt);

        void factorize(const BSRMatrix &A, CudaRuntime rt);

        void apply(ctd::span<const double> r, ctd::span<double> z, CudaRuntime rt);

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

        std::vector<LevelInfo> levels_;
        std::vector<int> level_node_offsets_;
        std::vector<int> level_matrix_offsets_;

        Buf<int> fine_ancestors_;
        Buf<int> level_node_offsets_device_;
        Buf<int> level_matrix_offsets_device_;
        Buf<int> fine_to_slot_;
        Buf<int> slot_to_fine_;
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
    };

} // namespace polysolve::linear::mas
