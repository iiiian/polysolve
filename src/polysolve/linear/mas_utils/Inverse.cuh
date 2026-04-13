#pragma once

#include <polysolve/linear/mas_utils/CudaUtils.cuh>
#include <polysolve/linear/mas_utils/SimpleLinalg.cuh>

namespace polysolve::linear::mas
{
    // Support dim: 1, 2, 3
    template <int D>
    __both__ void inverse(MatRef<const double, D, D> m, MatRef<double, D, D> out);

} // namespace polysolve::linear::mas
