#include "PCGSolver.hpp"

#include <Eigen/Core>

#include <cub/cub.cuh>

#include <cuda/algorithm>
#include <cuda/devices>
#include <cuda/stream>
#include <cuda/memory_pool>
#include <cuda/std/cmath>
#include <cuda/std/span>
#include <cuda/std/optional>

#include <stdexcept>
#include <string>

#include <polysolve/linear/mas_utils/BCOOMatrix.hpp>
#include <polysolve/linear/mas_utils/CudaUtils.cuh>
#include <polysolve/linear/mas_utils/InnerProduct.hpp>
#include <polysolve/linear/mas_utils/Inverse.cuh>
#include <polysolve/linear/mas_utils/SimpleLinalg.cuh>
#include <polysolve/linear/mas_utils/Spmv.hpp>

namespace polysolve::linear
{

    using namespace mas;

    namespace
    {
        void scalar_division(
            ctd::span<const double> num,
            ctd::span<const double> denom,
            ctd::span<double> out,
            CudaRuntime rt)
        {
            auto op = [num, denom, out] __device__(int) {
                out[0] = (ctd::abs(denom[0]) < 1e-20) ? 0.0 : (num[0] / denom[0]);
            };
            cub::DeviceFor::Bulk(1, op, rt.stream.get());
        }

        template <int D>
        void compute_diag_inv(BCOOView A, ctd::span<double> out, CudaRuntime rt)
        {
            auto diag_index = A.diag_index;
            auto vals = A.vals;
            auto op = [diag_index, vals, out] __device__(int row) {
                int idx = diag_index[row];
                auto inv_out = MatRef<double, D, D>::row_major(out.data() + row * D * D);

                // no diag
                if (idx == -1)
                {
                    assign(inv_out, Mat<double, D, D>::identity());
                    return;
                }

                auto diag_block = MatRef<const double, D, D>::row_major(vals.data() + idx * D * D);
                inverse<D>(diag_block, inv_out);
            };

            cub::DeviceFor::Bulk(diag_index.size(), op, rt.stream.get());
        }

        template <int D>
        void apply_precond(
            ctd::span<const double> diag_inv,
            ctd::span<const double> x,
            ctd::span<double> y,
            CudaRuntime rt)
        {
            auto op = [diag_inv, x, y] __device__(int idx) {
                auto inv_mat = MatRef<const double, D, D>::row_major(diag_inv.data() + idx * D * D);
                auto x_in = MatRef<const double, D, 1>::row_major(x.data() + idx * D);
                auto y_out = MatRef<double, D, 1>::row_major(y.data() + idx * D);

                assign(y_out, mat_mul(inv_mat, x_in));
            };

            cub::DeviceFor::Bulk(static_cast<int>(x.size() / D), op, rt.stream.get());
        }

        void apply_precond_dispatch(
            int block_dim,
            ctd::span<const double> diag_inv,
            ctd::span<const double> x,
            ctd::span<double> y,
            CudaRuntime rt)
        {
            switch (block_dim)
            {
            case 1:
                apply_precond<1>(diag_inv, x, y, rt);
                break;
            case 2:
                apply_precond<2>(diag_inv, x, y, rt);
                break;
            case 3:
                apply_precond<3>(diag_inv, x, y, rt);
                break;
            default:
                throw std::runtime_error("[CudaPCG] Unsupported block size.");
            }
        }

        void axpby(
            double h_alpha,
            const double *d_alpha,
            double h_beta,
            const double *d_beta,
            ctd::span<const double> x,
            ctd::span<double> y,
            CudaRuntime rt)
        {
            auto op = [h_alpha, d_alpha, h_beta, d_beta, x, y] __device__(int idx) {
                double alpha = h_alpha * ((d_alpha == nullptr) ? 1.0 : *d_alpha);
                double beta = h_beta * ((d_beta == nullptr) ? 1.0 : *d_beta);
                y[idx] = alpha * x[idx] + beta * y[idx];
            };
            cub::DeviceFor::Bulk(x.size(), op, rt.stream.get());
        }
    } // namespace

    class CudaPCG::CudaPCGImpl
    {
    private:
        int block_dim_ = 1;
        int max_iter_ = 1e5;
        int true_residual_period_ = 50;
        double abs_tol_ = 1e-20;
        double rel_tol_ = 1e-6;
        bool use_preconditioned_residual_norm_ = true;

        int dim_ = 0;
        int padded_dim_ = 0;
        int iterations_ = 0;
        double residual_norm_ = 0.0;
        CudaPCGStatus status_ = CudaPCGStatus::Running;

        BCOOMatrix A_;
        Buf<double> diag_inv_;
        Buf<double> x_;
        Buf<double> b_;
        Buf<double> r_;
        Buf<double> p_;
        Buf<double> z_;
        Buf<double> Ap_;
        Buf<char> reduction_storage_;
        Buf<double> scalar_rz_;
        Buf<double> scalar_pAp_;
        Buf<double> scalar_alpha_;
        Buf<double> scalar_beta_;
        Buf<double> scalar_rz_old_;
        Buf<double> scalar_rr_;

        ctd::optional<cu::device_ref> default_device_;
        ctd::optional<cu::stream> default_stream_;
        ctd::optional<cu::device_memory_pool> default_mem_pool_;

    public:
        CudaPCGImpl()
        {
            if (cu::devices.size() == 0)
            {
                throw std::runtime_error("No Nvidia GPU!!");
            }

            default_device_.emplace(cu::devices[0]);
            default_stream_.emplace(*default_device_);
            default_mem_pool_.emplace(*default_device_);
        }

        void set_parameters(const json &params)
        {
            if (params.contains("max_iter"))
                max_iter_ = params["max_iter"];
            if (params.contains("relative_tolerance"))
                rel_tol_ = params["relative_tolerance"];
            if (params.contains("absolute_tolerance"))
                abs_tol_ = params["absolute_tolerance"];
            if (params.contains("true_residual_period"))
                true_residual_period_ = params["true_residual_period"];
            if (params.contains("use_preconditioned_residual_norm"))
                use_preconditioned_residual_norm_ = params["use_preconditioned_residual_norm"];
        }

        void get_info(json &params) const
        {
            params["solver_iter"] = iterations_;
            params["solver_error"] = residual_norm_;
            params["solver_status"] = pcg_status_to_string(status_);
        }

        void analyze_pattern(const StiffnessMatrix &, const int) {}

        void factorize(const StiffnessMatrix &A)
        {
            CudaRuntime rt{*default_stream_, default_mem_pool_->as_ref()};
            A_ = BCOOMatrix{A, block_dim_, rt};

            BCOOView view = A_.view();
            dim_ = A.rows();
            padded_dim_ = view.block_dim * view.dim;

            diag_inv_ = cu::make_buffer<double>(
                rt.stream,
                rt.mr,
                view.block_dim * view.block_dim * view.dim,
                cu::no_init);

            switch (view.block_dim)
            {
            case 1:
                compute_diag_inv<1>(view, *diag_inv_, rt);
                break;
            case 2:
                compute_diag_inv<2>(view, *diag_inv_, rt);
                break;
            case 3:
                compute_diag_inv<3>(view, *diag_inv_, rt);
                break;
            default:
                throw std::runtime_error("[CudaPCG] Unsupported block size.");
            }

            x_ = cu::make_buffer<double>(rt.stream, rt.mr, padded_dim_, cu::no_init);
            b_ = cu::make_buffer<double>(rt.stream, rt.mr, padded_dim_, cu::no_init);
            r_ = cu::make_buffer<double>(rt.stream, rt.mr, padded_dim_, cu::no_init);
            p_ = cu::make_buffer<double>(rt.stream, rt.mr, padded_dim_, cu::no_init);
            z_ = cu::make_buffer<double>(rt.stream, rt.mr, padded_dim_, cu::no_init);
            Ap_ = cu::make_buffer<double>(rt.stream, rt.mr, padded_dim_, cu::no_init);

            scalar_rz_ = cu::make_buffer<double>(rt.stream, rt.mr, 1, cu::no_init);
            scalar_pAp_ = cu::make_buffer<double>(rt.stream, rt.mr, 1, cu::no_init);
            scalar_alpha_ = cu::make_buffer<double>(rt.stream, rt.mr, 1, cu::no_init);
            scalar_beta_ = cu::make_buffer<double>(rt.stream, rt.mr, 1, cu::no_init);
            scalar_rz_old_ = cu::make_buffer<double>(rt.stream, rt.mr, 1, cu::no_init);
            scalar_rr_ = cu::make_buffer<double>(rt.stream, rt.mr, 1, cu::no_init);

            rt.stream.sync();
        }

        void solve(const Eigen::Ref<const Eigen::VectorXd> b, Eigen::Ref<Eigen::VectorXd> x)
        {
            status_ = CudaPCGStatus::Running;

            if (b.size() != x.size() || !check_buffer_size(b.size()))
            {
                throw std::runtime_error("[CudaPCG] Size mismatch. Did you forget to call factorize?");
            }

            CudaRuntime rt{*default_stream_, default_mem_pool_->as_ref()};

            cu::copy_bytes(
                rt.stream,
                ctd::span<const double>(b.data(), dim_),
                ctd::span<double>(b_->data(), dim_));
            cu::fill_bytes(
                rt.stream,
                ctd::span<double>(b_->data() + dim_, padded_dim_ - dim_),
                0);

            // The solver sometimes fails to converge if we use input x as initial value.
            // Maybe the caller does not initialize x properly?
            // Set initial x to zero to work around this issue for now.
            cu::fill_bytes(rt.stream, *x_, 0);

            pcg_solve(rt);

            cu::copy_bytes(
                rt.stream,
                ctd::span<const double>(x_->data(), dim_),
                ctd::span<double>(x.data(), dim_));
            rt.stream.sync();
        }

        bool check_buffer_size(int n) const
        {
            if (n <= 0 || !diag_inv_ || !x_ || !b_ || !r_ || !p_ || !z_ || !Ap_
                || !scalar_rz_ || !scalar_pAp_ || !scalar_alpha_ || !scalar_beta_
                || !scalar_rz_old_ || !scalar_rr_)
            {
                return false;
            }

            BCOOView view = A_.view();
            int block_n = (n + view.block_dim - 1) / view.block_dim;
            int block_size = view.block_dim * view.block_dim;

            if (n != dim_ || block_n != view.dim)
            {
                return false;
            }

            if (view.rows.size() != view.non_zeros
                || view.cols.size() != view.non_zeros
                || view.vals.size() != block_size * view.non_zeros
                || view.diag_index.size() != view.dim
                || diag_inv_->size() != block_size * view.dim)
            {
                return false;
            }

            if (x_->size() != padded_dim_
                || b_->size() != padded_dim_
                || r_->size() != padded_dim_
                || p_->size() != padded_dim_
                || z_->size() != padded_dim_
                || Ap_->size() != padded_dim_)
            {
                return false;
            }

            if (scalar_rz_->size() < 1
                || scalar_pAp_->size() < 1
                || scalar_alpha_->size() < 1
                || scalar_beta_->size() < 1
                || scalar_rz_old_->size() < 1
                || scalar_rr_->size() < 1)
            {
                return false;
            }

            return true;
        }

        void pcg_solve(CudaRuntime rt)
        {
            BCOOView view = A_.view();

            // Compute initial residual r = b-Ax.
            spmv(view, *x_, *r_, rt);
            axpby(1.0, nullptr, -1.0, nullptr, *b_, *r_, rt);

            // Compute z = M^-1 r.
            apply_precond_dispatch(view.block_dim, *diag_inv_, *r_, *z_, rt);
            // Initial search direction p = z;
            cu::copy_bytes(rt.stream, *z_, *p_);

            // Compute rz = r^T M^-1 r.
            inner_product(*r_, *z_, *scalar_rz_, rt);
            const double rz0 = device2host(scalar_rz_->data(), rt);
            if (ctd::isnan(rz0) || !ctd::isfinite(rz0))
            {
                throw std::runtime_error("[CudaPCG] Invalid initial residual.");
            }

            // Compute rr = r^T r.
            double rr0 = 0.0;
            if (!use_preconditioned_residual_norm_)
            {
                inner_product(*r_, *r_, *scalar_rr_, rt);
                rr0 = device2host(scalar_rr_->data(), rt);
            }

            for (int k = 1; k <= max_iter_; ++k)
            {
                // Compute Ap = A p.
                spmv(view, *p_, *Ap_, rt);
                // Compute pAp = p^T * A * p.
                inner_product(*p_, *Ap_, *scalar_pAp_, rt);
                // Compute alpha = (r M^-1 r) / (p^T A p).
                scalar_division(*scalar_rz_, *scalar_pAp_, *scalar_alpha_, rt);
                // Compute x = x + alpha A p.
                axpby(1.0, scalar_alpha_->data(), 1.0, nullptr, *p_, *x_, rt);

                // Compute residual b-Ax directly.
                if (k % true_residual_period_ == 0)
                {
                    spmv(view, *x_, *r_, rt);
                    axpby(1.0, nullptr, -1.0, nullptr, *b_, *r_, rt);
                }
                // Compute residual update using r' = r - alpha A p.
                // This saves one spmv but accumulates floating point error overtime.
                else
                {
                    axpby(-1.0, scalar_alpha_->data(), 1.0, nullptr, *Ap_, *r_, rt);
                }

                // Compute z = M^-1 r.
                // TODO: MAS Precond
                apply_precond_dispatch(view.block_dim, *diag_inv_, *r_, *z_, rt);

                // Compute rz = r M^-1 r.
                cu::copy_bytes(rt.stream, *scalar_rz_, *scalar_rz_old_);
                inner_product(*r_, *z_, *scalar_rz_, rt);

                iterations_ = k;
                bool converged = false;

                // Check convergence every 10 iterations.
                if (k % 10 == 0)
                {
                    if (use_preconditioned_residual_norm_)
                    {
                        double rz_new = device2host(scalar_rz_->data(), rt);
                        residual_norm_ = ctd::sqrt(rz_new);
                        if (rz_new <= rel_tol_ * rel_tol_ * rz0 || rz_new <= abs_tol_ * abs_tol_)
                        {
                            status_ = (rz_new <= abs_tol_ * abs_tol_)
                                          ? CudaPCGStatus::ReachAbsoluteTolerance
                                          : CudaPCGStatus::ReachRelativeTolerance;
                            converged = true;
                        }
                    }
                    else
                    {
                        inner_product(*r_, *r_, *scalar_rr_, rt);
                        double rr = device2host(scalar_rr_->data(), rt);
                        residual_norm_ = ctd::sqrt(rr);
                        if (rr <= rel_tol_ * rel_tol_ * rr0 || rr <= abs_tol_ * abs_tol_)
                        {
                            status_ = (rr <= abs_tol_ * abs_tol_)
                                          ? CudaPCGStatus::ReachAbsoluteTolerance
                                          : CudaPCGStatus::ReachRelativeTolerance;
                            converged = true;
                        }
                    }
                }

                if (converged)
                {
                    break;
                }

                // Compute beta = rz / rz_old.
                scalar_division(*scalar_rz_, *scalar_rz_old_, *scalar_beta_, rt);
                // Compute direction update p' = M^-1 r + beta p.
                axpby(1.0, nullptr, 1.0, scalar_beta_->data(), *z_, *p_, rt);
            }

            if (iterations_ == max_iter_)
            {
                status_ = CudaPCGStatus::ReachMaxIterations;
            }
        }
    };

    CudaPCG::CudaPCG()
        : impl_(std::make_unique<CudaPCGImpl>())
    {
    }

    CudaPCG::~CudaPCG() = default;

    void CudaPCG::set_parameters(const json &params)
    {
        const std::string solver_name = name();
        if (!params.contains(solver_name))
        {
            return;
        }

        impl_->set_parameters(params[solver_name]);
    }

    void CudaPCG::get_info(json &params) const
    {
        impl_->get_info(params);
    }

    void CudaPCG::analyze_pattern(const StiffnessMatrix &A, const int precond_num)
    {
        impl_->analyze_pattern(A, precond_num);
    }

    void CudaPCG::factorize(const StiffnessMatrix &A)
    {
        impl_->factorize(A);
    }

    void CudaPCG::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        impl_->solve(b, x);
    }

    std::string CudaPCG::name() const
    {
        return "CUDA_PCG";
    }

} // namespace polysolve::linear
