#pragma once

#include <polysolve/linear/mas_utils/BSRMatrix.hpp>
#include <polysolve/linear/mas_utils/CudaUtils.cuh>

#include <array>

namespace polysolve::linear::mas
{
    constexpr int MAS_MAX_COARSE_LEVEL = 4;

    struct PaddedTopology
    {
        Buf<int> real_to_padded;
        Buf<int> padded_to_real;
        Buf<int> rows;
        Buf<int> cols;
        int node_num = 0;
        int padded_node_num = 0;
    };

    struct CoarseSpace
    {
        Buf<int> map;
        std::array<int, MAS_MAX_COARSE_LEVEL> cco_nums{};
    };

    struct CoarseMatrices
    {
        Buf<double> data;
        std::array<int, MAS_MAX_COARSE_LEVEL> matrix_offsets{};
        std::array<int, MAS_MAX_COARSE_LEVEL> matrix_counts{};
        int mat_dim = 0;
        int mat_storage_size = 0;
        int total_matrix_num = 0;
    };

    struct MASWorkspace
    {
        Buf<double> multi_level_r;
        Buf<double> multi_level_z;
        std::array<int, MAS_MAX_COARSE_LEVEL> level_offsets{};
        std::array<int, MAS_MAX_COARSE_LEVEL> level_sizes{};
        int total_level_nodes = 0;
    };

    class MASPreconditioner
    {
    public:
        static constexpr int MAX_COARSE_LEVEL = MAS_MAX_COARSE_LEVEL;

        bool empty() const
        {
            return !initialized_;
        }

        void factorize(const BSRMatrix &A, ctd::span<const int> part_offsets, CudaRuntime rt);
        void apply(ctd::span<const double> r, ctd::span<double> z, CudaRuntime rt);

    private:
        bool initialized_ = false;
        int vector_size_ = 0;
        int padded_vector_size_ = 0;
        int block_dim_ = 0;
        PaddedTopology padded_topology_;
        CoarseSpace coarse_space_;
        CoarseMatrices coarse_matrices_;
        MASWorkspace workspace_;
    };
} // namespace polysolve::linear::mas
