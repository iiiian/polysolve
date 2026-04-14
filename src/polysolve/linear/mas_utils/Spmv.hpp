#pragma once

#include <polysolve/linear/mas_utils/CudaUtils.cuh>
#include <polysolve/linear/mas_utils/BCOOMatrix.hpp>
#include <cuda/std/span>

namespace polysolve::linear::mas
{
    /// @brief Compute y = Ax. Does not sync implicitly.
    void spmv(BCOOView A, ctd::span<const double> x, ctd::span<double> y, CudaRuntime rt);

} // namespace polysolve::linear::mas
