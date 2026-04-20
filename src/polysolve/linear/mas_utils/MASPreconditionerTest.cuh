#pragma once

#include <polysolve/linear/mas_utils/CudaUtils.cuh>

namespace polysolve::linear::mas
{
    void invert_packed_matrices_for_test(ctd::span<double> mats, int block_dim, CudaRuntime rt);
    void apply_packed_matrices_for_test(
        ctd::span<const double> mats,
        ctd::span<const double> x,
        ctd::span<double> y,
        int block_dim,
        CudaRuntime rt);
}
