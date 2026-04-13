#include <polysolve/linear/mas_utils/Inverse.cuh>

#include <cuda/std/cmath>
#include <polysolve/linear/mas_utils/CudaUtils.cuh>

namespace polysolve::linear::mas
{

    template <>
    __both__ void inverse<1>(MatRef<const double, 1, 1> m, MatRef<double, 1, 1> out)
    {
        if (ctd::abs(m(0)) < 1e-20)
        {
            // Falling back to identity is a good default for preconditioner.
            assign(out, Mat<double, 1, 1>::identity());
            out(0) = 1.0;
        }
        else
        {
            out(0) = 1.0 / m(0);
        }
    }

    template <>
    __both__ void inverse<2>(MatRef<const double, 2, 2> m, MatRef<double, 2, 2> out)
    {
        double a00 = m(0, 0);
        double a01 = m(0, 1);
        double a10 = m(1, 0);
        double a11 = m(1, 1);

        double det = a00 * a11 - a01 * a10;
        if (ctd::abs(det) < 1e-20)
        {
            // Falling back to identity is a good default for preconditioner.
            assign(out, Mat<double, 2, 2>::identity());
        }
        else
        {
            double inv_det = 1.0 / det;
            out(0, 0) = a11 * inv_det;
            out(0, 1) = -a01 * inv_det;
            out(1, 0) = -a10 * inv_det;
            out(1, 1) = a00 * inv_det;
        }
    }

    template <>
    __both__ void inverse<3>(MatRef<const double, 3, 3> m, MatRef<double, 3, 3> out)
    {
        double a00 = m(0, 0);
        double a01 = m(0, 1);
        double a02 = m(0, 2);
        double a10 = m(1, 0);
        double a11 = m(1, 1);
        double a12 = m(1, 2);
        double a20 = m(2, 0);
        double a21 = m(2, 1);
        double a22 = m(2, 2);

        double c00 = a11 * a22 - a12 * a21;
        double c01 = a02 * a21 - a01 * a22;
        double c02 = a01 * a12 - a02 * a11;
        double c10 = a12 * a20 - a10 * a22;
        double c11 = a00 * a22 - a02 * a20;
        double c12 = a02 * a10 - a00 * a12;
        double c20 = a10 * a21 - a11 * a20;
        double c21 = a01 * a20 - a00 * a21;
        double c22 = a00 * a11 - a01 * a10;

        double det = a00 * c00 + a01 * c10 + a02 * c20;
        if (ctd::abs(det) < 1e-20)
        {
            // Falling back to identity is a good default for preconditioner.
            assign(out, Mat<double, 3, 3>::identity());
        }
        else
        {
            double inv_det = 1.0 / det;
            out(0, 0) = c00 * inv_det;
            out(0, 1) = c01 * inv_det;
            out(0, 2) = c02 * inv_det;
            out(1, 0) = c10 * inv_det;
            out(1, 1) = c11 * inv_det;
            out(1, 2) = c12 * inv_det;
            out(2, 0) = c20 * inv_det;
            out(2, 1) = c21 * inv_det;
            out(2, 2) = c22 * inv_det;
        }
    }

} // namespace polysolve::linear::mas
