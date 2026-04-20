#pragma once

#include <polysolve/linear/mas_utils/BSRMatrix.hpp>
#include <polysolve/linear/mas_utils/CudaUtils.cuh>

namespace polysolve::linear::mas
{
    class MASPreconditioner
    {
    public:
        bool empty() const
        {
            return !initialized_;
        }

        void factorize(const BSRMatrix &A, CudaRuntime rt);
        void apply(ctd::span<const double> r, ctd::span<double> z, CudaRuntime rt);

    private:
        bool initialized_ = false;
        int vector_size_ = 0;
    };
} // namespace polysolve::linear::mas
