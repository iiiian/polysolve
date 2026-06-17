#include "ForcingTermStrategy.hpp"

#include <polysolve/Utils.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace polysolve::nonlinear
{
    namespace
    {
        template <typename T>
        T param_or(const json &params, const std::string &name, const T &default_value)
        {
            if (params.contains(name))
                return params[name].get<T>();

            return default_value;
        }

        json extract_forcing_term_params(const std::string &key, const json &params)
        {
            const json &solver_params = (params.find(key) != params.end()) ? params[key] : params;
            if (solver_params.contains("forcing_term_strategy"))
                return solver_params["forcing_term_strategy"];
            return json{{"type", "An_Mo"}};
        }

        class NoForcingTermStrategy final : public ForcingTermStrategy
        {
        public:
            bool is_adaptive() const override { return false; }

            double begin_iteration(
                const TVector &,
                const Problem &,
                const NormType) override
            {
                return 0;
            }

            void record_linear_model(
                const TVector &,
                const TVector &) override {}

            void accept_step(const double) override {}
        };

        class AdaptiveForcingTermStrategy : public ForcingTermStrategy
        {
        public:
            bool is_adaptive() const final { return true; }

            double begin_iteration(
                const TVector &current_F,
                const Problem &problem,
                const NormType norm_type) final
            {
                if (accepted_observation_)
                {
                    const double previous_norm = problem.grad_norm(accepted_observation_->old_F, norm_type);
                    const double current_norm = problem.grad_norm(current_F, norm_type);
                    const double model_residual_norm = problem.grad_norm(accepted_observation_->accepted_model_residual, norm_type);
                    update_eta(previous_norm, current_norm, model_residual_norm);
                    accepted_observation_.reset();
                }

                current_model_.reset();
                return eta_;
            }

            void record_linear_model(
                const TVector &old_F,
                const TVector &full_step_model_residual) final
            {
                current_model_ = CurrentLinearModel{old_F, full_step_model_residual};
            }

            void accept_step(const double alpha) final
            {
                if (!current_model_ || !std::isfinite(alpha) || alpha <= 0)
                {
                    current_model_.reset();
                    return;
                }

                accepted_observation_ = AcceptedObservation{
                    current_model_->old_F,
                    (1.0 - alpha) * current_model_->old_F + alpha * current_model_->full_step_model_residual};
                current_model_.reset();
            }

        protected:
            explicit AdaptiveForcingTermStrategy(const double initial_eta)
                : eta_(initial_eta)
            {
            }

            virtual void update_eta(
                const double previous_norm,
                const double current_norm,
                const double model_residual_norm) = 0;

            double eta_;

        private:
            struct CurrentLinearModel
            {
                TVector old_F;
                TVector full_step_model_residual;
            };

            struct AcceptedObservation
            {
                TVector old_F;
                TVector accepted_model_residual;
            };

            std::optional<CurrentLinearModel> current_model_;
            std::optional<AcceptedObservation> accepted_observation_;
        };

        class AnMoForcingTermStrategy final : public AdaptiveForcingTermStrategy
        {
        public:
            AnMoForcingTermStrategy(const json &params, spdlog::logger &logger)
                : AdaptiveForcingTermStrategy(param_or<double>(params, "initial_eta", 0.5))
            {
                p1_ = param_or<double>(params, "p1", 0.1);
                p2_ = param_or<double>(params, "p2", 0.4);
                p3_ = param_or<double>(params, "p3", 0.7);
                shrink_moderate_ = param_or<double>(params, "shrink_moderate", 0.8);
                shrink_strong_ = param_or<double>(params, "shrink_strong", 0.5);
                safeguard_threshold_ = param_or<double>(params, "safeguard_threshold", 0.1);

                if (eta_ <= 0 || eta_ >= 1)
                    log_and_throw_error(logger, "An_Mo forcing term initial_eta must satisfy 0 < initial_eta < 1, instead got {}", eta_);
                if (p1_ <= 0 || p1_ >= 0.5)
                    log_and_throw_error(logger, "An_Mo forcing term p1 must satisfy 0 < p1 < 0.5, instead got {}", p1_);
                if (!(p1_ < p2_ && p2_ < p3_ && p3_ < 1))
                    log_and_throw_error(logger, "An_Mo forcing term thresholds must satisfy 0 < p1 < p2 < p3 < 1, instead got p1={} p2={} p3={}", p1_, p2_, p3_);
                if (shrink_moderate_ <= 0 || shrink_moderate_ >= 1)
                    log_and_throw_error(logger, "An_Mo forcing term shrink_moderate must satisfy 0 < shrink_moderate < 1, instead got {}", shrink_moderate_);
                if (shrink_strong_ <= 0 || shrink_strong_ >= 1)
                    log_and_throw_error(logger, "An_Mo forcing term shrink_strong must satisfy 0 < shrink_strong < 1, instead got {}", shrink_strong_);
                if (safeguard_threshold_ <= 0 || safeguard_threshold_ >= 1)
                    log_and_throw_error(logger, "An_Mo forcing term safeguard_threshold must satisfy 0 < safeguard_threshold < 1, instead got {}", safeguard_threshold_);
            }

        private:
            void update_eta(
                const double previous_norm,
                const double current_norm,
                const double model_residual_norm) override
            {
                if (previous_norm <= 0 || !std::isfinite(previous_norm)
                    || !std::isfinite(current_norm) || !std::isfinite(model_residual_norm))
                    return;

                const double actual_reduction = previous_norm - current_norm;
                const double predicted_reduction = previous_norm - model_residual_norm;
                if (!std::isfinite(actual_reduction) || !std::isfinite(predicted_reduction) || predicted_reduction <= 0)
                    return;

                const double ratio = actual_reduction / predicted_reduction;
                if (!std::isfinite(ratio))
                    return;

                const double eta_used = eta_;
                double next_eta = eta_used;
                if (ratio < p1_)
                    next_eta = 1.0 - 2.0 * p1_;
                else if (ratio < p2_)
                    next_eta = eta_used;
                else if (ratio < p3_)
                    next_eta = shrink_moderate_ * eta_used;
                else
                    next_eta = shrink_strong_ * eta_used;

                if (has_previous_ratio_
                    && previous_eta_ > safeguard_threshold_
                    && eta_used > safeguard_threshold_
                    && previous_ratio_ < p1_
                    && ratio < p1_)
                {
                    next_eta = shrink_strong_ * eta_used;
                }

                previous_eta_ = eta_used;
                previous_ratio_ = ratio;
                has_previous_ratio_ = true;
                eta_ = next_eta;
            }

            double p1_;
            double p2_;
            double p3_;
            double shrink_moderate_;
            double shrink_strong_;
            double safeguard_threshold_;
            bool has_previous_ratio_ = false;
            double previous_ratio_ = std::numeric_limits<double>::quiet_NaN();
            double previous_eta_ = std::numeric_limits<double>::quiet_NaN();
        };

        class EisenstatForcingTermStrategy final : public AdaptiveForcingTermStrategy
        {
        public:
            EisenstatForcingTermStrategy(const json &params, spdlog::logger &logger)
                : AdaptiveForcingTermStrategy(param_or<double>(params, "initial_eta", 0.5))
            {
                eta_max_ = param_or<double>(params, "eta_max", 0.9);
                gamma_ = param_or<double>(params, "gamma", 0.9);
                alpha_ = param_or<double>(params, "alpha", 2.0);
                safeguard_threshold_ = param_or<double>(params, "safeguard_threshold", 0.1);

                if (eta_ < 0 || eta_ >= 1)
                    log_and_throw_error(logger, "Eisenstat forcing term initial_eta must satisfy 0 <= initial_eta < 1, instead got {}", eta_);
                if (eta_max_ <= 0 || eta_max_ >= 1)
                    log_and_throw_error(logger, "Eisenstat forcing term eta_max must satisfy 0 < eta_max < 1, instead got {}", eta_max_);
                if (gamma_ <= 0 || gamma_ > 1)
                    log_and_throw_error(logger, "Eisenstat forcing term gamma must satisfy 0 < gamma <= 1, instead got {}", gamma_);
                if (alpha_ <= 1 || alpha_ > 2)
                    log_and_throw_error(logger, "Eisenstat forcing term alpha must satisfy 1 < alpha <= 2, instead got {}", alpha_);
                if (safeguard_threshold_ <= 0 || safeguard_threshold_ >= 1)
                    log_and_throw_error(logger, "Eisenstat forcing term safeguard_threshold must satisfy 0 < safeguard_threshold < 1, instead got {}", safeguard_threshold_);
            }

        private:
            void update_eta(
                const double previous_norm,
                const double current_norm,
                const double) override
            {
                if (previous_norm <= 0 || current_norm < 0
                    || !std::isfinite(previous_norm) || !std::isfinite(current_norm))
                    return;

                const double theta = gamma_ * std::pow(current_norm / previous_norm, alpha_);
                const double safeguard = gamma_ * std::pow(eta_, alpha_);
                if (!std::isfinite(theta) || !std::isfinite(safeguard))
                    return;

                const double next_eta = (safeguard <= safeguard_threshold_)
                                            ? theta
                                            : std::max(theta, safeguard);
                eta_ = std::min(next_eta, eta_max_);
            }

            double eta_max_;
            double gamma_;
            double alpha_;
            double safeguard_threshold_;
        };
    } // namespace

    std::unique_ptr<ForcingTermStrategy> ForcingTermStrategy::create(
        const std::string &param_key,
        const json &solver_params,
        spdlog::logger &logger)
    {
        const json strategy_params = extract_forcing_term_params(param_key, solver_params);
        const std::string strategy_type = param_or<std::string>(strategy_params, "type", "An_Mo");
        const json strategy_options =
            (strategy_params.contains(strategy_type) && !strategy_params[strategy_type].is_null())
                ? strategy_params[strategy_type]
                : json::object();

        if (strategy_type == "An_Mo")
            return std::make_unique<AnMoForcingTermStrategy>(strategy_options, logger);
        if (strategy_type == "Eisenstat")
            return std::make_unique<EisenstatForcingTermStrategy>(strategy_options, logger);
        if (strategy_type == "None")
            return std::make_unique<NoForcingTermStrategy>();

        log_and_throw_error(logger, "Unknown forcing_term_strategy type {}", strategy_type);
        return nullptr;
    }
} // namespace polysolve::nonlinear
