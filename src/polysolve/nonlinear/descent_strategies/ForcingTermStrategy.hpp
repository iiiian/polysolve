#pragma once

#include <polysolve/Utils.hpp>
#include <polysolve/nonlinear/Problem.hpp>

#include <memory>
#include <string>

namespace polysolve::nonlinear
{
    class ForcingTermStrategy
    {
    public:
        using TVector = Problem::TVector;

        virtual ~ForcingTermStrategy() = default;

        static std::unique_ptr<ForcingTermStrategy> create(
            const std::string &param_key,
            const json &solver_params,
            spdlog::logger &logger);

        virtual bool is_adaptive() const = 0;

        /// Update the forcing term from the previously accepted step, if any,
        /// and return the forcing term eta for the current Newton linear solve.
        virtual double begin_iteration(
            const TVector &current_F,
            const Problem &problem,
            const NormType norm_type) = 0;

        /// Record the model residual F_k + J_k s_k for the just-computed
        /// full Newton step. The accepted line-search step is not known yet.
        virtual void record_linear_model(
            const TVector &old_F,
            const TVector &full_step_model_residual) = 0;

        /// Accept the current linear model with line-search step size alpha.
        virtual void accept_step(const double alpha) = 0;
    };
} // namespace polysolve::nonlinear
