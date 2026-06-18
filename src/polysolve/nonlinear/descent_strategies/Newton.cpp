#include "Newton.hpp"

#include "ForcingTermStrategy.hpp"

#include <polysolve/Utils.hpp>

#if defined(SPDLOG_FMT_EXTERNAL)
#include <fmt/color.h>
#else
#include <spdlog/fmt/bundled/color.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>

namespace polysolve::nonlinear
{
    namespace
    {
        void copy_newton_param(const json &src, json &dst, const std::string &dst_key, const std::string &name)
        {
            dst[dst_key][name] = src["Newton"][name];
        }

        void log_linear_solver_info(spdlog::logger &logger, const std::string &solver_name, const json &info)
        {
            if (!info.contains("num_iterations") || !info.contains("final_res_norm") || !info.contains("converged"))
            {
                return;
            }
            if (!info["num_iterations"].is_number_integer() || !info["final_res_norm"].is_number() || !info["converged"].is_boolean())
            {
                return;
            }

            logger.info(
                "[Newton] linear solver {}: iterations={} residual={:.6e} converged={}",
                solver_name,
                info["num_iterations"].get<long long>(),
                info["final_res_norm"].get<double>(),
                info["converged"].get<bool>());
        }
    } // namespace

    std::vector<std::shared_ptr<DescentStrategy>> Newton::create_solver(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger,
        const NormType norm_type)
    {
        // Copies stuff from main newton
        json proj_solver_params = R"({"ProjectedNewton": {}})"_json;
        json reg_solver_params = R"({"RegularizedNewton": {}})"_json;

        for (const char *name : {
                 "residual_tolerance",
                 "forcing_term_strategy"})
        {
            copy_newton_param(solver_params, proj_solver_params, "ProjectedNewton", name);
            copy_newton_param(solver_params, reg_solver_params, "RegularizedNewton", name);
        }

        reg_solver_params["RegularizedNewton"]["reg_weight_min"] = solver_params["Newton"]["reg_weight_min"];
        reg_solver_params["RegularizedNewton"]["reg_weight_max"] = solver_params["Newton"]["reg_weight_max"];
        reg_solver_params["RegularizedNewton"]["reg_weight_inc"] = solver_params["Newton"]["reg_weight_inc"];

        std::vector<std::shared_ptr<DescentStrategy>> res;
        const bool force_psd_projection = solver_params["Newton"]["force_psd_projection"];
        if (!force_psd_projection)
            res.push_back(std::make_unique<Newton>(
                sparse,
                solver_params, linear_solver_params,
                characteristic_length, logger, norm_type));

        const bool use_psd_projection = solver_params["Newton"]["use_psd_projection"];
        if (use_psd_projection)
            res.push_back(std::make_unique<ProjectedNewton>(
                sparse,
                proj_solver_params, linear_solver_params,
                characteristic_length, logger, norm_type));

        const double reg_weight_min = solver_params["Newton"]["reg_weight_min"];
        if (reg_weight_min > 0)
            res.push_back(std::make_unique<RegularizedNewton>(
                sparse, solver_params["Newton"]["use_psd_projection_in_regularized"],
                reg_solver_params, linear_solver_params,
                characteristic_length, logger, norm_type));

        if (res.empty())
            log_and_throw_error(logger, "Newton needs to have at least one of force_psd_projection=false, reg_weight_min>0, or use_psd_projection=true");

        return res;
    }

    Newton::Newton(const bool sparse,
                   const double residual_tolerance,
                   const std::string &param_key,
                   const json &solver_params,
                   const json &linear_solver_params,
                   const double characteristic_length,
                   spdlog::logger &logger,
                   const NormType norm_type)
        : Superclass(solver_params, characteristic_length, logger),
          is_sparse(sparse), characteristic_length(characteristic_length), residual_tolerance(residual_tolerance), norm_type(norm_type),
          forcing_term_param_key(param_key), forcing_term_solver_params(solver_params)
    {
        linear_solver = polysolve::linear::Solver::create(linear_solver_params, logger);

        if (linear_solver->is_dense() == sparse)
            log_and_throw_error(logger, "Newton linear solver must be {}, instead got {}", sparse ? "sparse" : "dense", linear_solver->name());

        if (residual_tolerance <= 0)
            log_and_throw_error(logger, "Newton residual_tolerance must be > 0, instead got {}", residual_tolerance);

        create_forcing_term_strategy();
    }

    Newton::~Newton() = default;

    Newton::Newton(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger,
        const NormType norm_type)
        : Newton(sparse, extract_param("Newton", "residual_tolerance", solver_params), "Newton", solver_params, linear_solver_params, characteristic_length, logger, norm_type)
    {
    }

    ProjectedNewton::ProjectedNewton(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger,
        const NormType norm_type)
        : Superclass(sparse, extract_param("ProjectedNewton", "residual_tolerance", solver_params), "ProjectedNewton", solver_params, linear_solver_params, characteristic_length, logger, norm_type)
    {
    }

    RegularizedNewton::RegularizedNewton(
        const bool sparse,
        const bool project_to_psd,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger,
        const NormType norm_type)
        : Superclass(sparse, extract_param("RegularizedNewton", "residual_tolerance", solver_params), "RegularizedNewton", solver_params, linear_solver_params, characteristic_length, logger, norm_type),
          project_to_psd(project_to_psd)
    {
        reg_weight_min = extract_param("RegularizedNewton", "reg_weight_min", solver_params);
        reg_weight_max = extract_param("RegularizedNewton", "reg_weight_max", solver_params);
        reg_weight_inc = extract_param("RegularizedNewton", "reg_weight_inc", solver_params);

        reg_weight = reg_weight_min;

        if (reg_weight_min <= 0)
            log_and_throw_error(logger, "Newton reg_weight_min must be  > 0, instead got {}", reg_weight_min);

        if (reg_weight_inc <= 1)
            log_and_throw_error(logger, "Newton reg_weight_inc must be  > 1, instead got {}", reg_weight_inc);

        if (reg_weight_max <= reg_weight_min)
            log_and_throw_error(logger, "Newton reg_weight_max must be  > {}, instead got {}", reg_weight_min, reg_weight_max);
    }

    // =======================================================================

    void Newton::create_forcing_term_strategy()
    {
        forcing_term_strategy = ForcingTermStrategy::create(forcing_term_param_key, forcing_term_solver_params, m_logger);
    }

    bool Newton::use_adaptive_forcing_term() const
    {
        return linear_solver->is_iterative() && forcing_term_strategy->is_adaptive();
    }

    double Newton::compute_residual_target(const double grad_norm, const double eta) const
    {
        if (!use_adaptive_forcing_term() || grad_norm <= 0 || !std::isfinite(grad_norm) || !std::isfinite(eta))
            return residual_tolerance;

        return std::max(residual_tolerance, eta * grad_norm);
    }

    // =======================================================================

    void Newton::reset(const int ndof)
    {
        Superclass::reset(ndof);
        internal_solver_info = json::array();
        create_forcing_term_strategy();
    }

    void RegularizedNewton::reset(const int ndof)
    {
        Superclass::reset(ndof);
        reg_weight = reg_weight_min;
    }

    void Newton::post_step(
        const TVector &,
        const TVector &,
        const TVector &,
        const double step_size,
        const TVector &)
    {
        forcing_term_strategy->accept_step(step_size);
    }

    // =======================================================================

    bool Newton::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        const double grad_norm = objFunc.grad_norm(grad, norm_type);
        const double eta = forcing_term_strategy->begin_iteration(grad, objFunc, norm_type);
        const double current_residual_tolerance = compute_residual_target(grad_norm, eta);

        const double residual =
            is_sparse ? solve_sparse_linear_system(objFunc, x, grad, direction, current_residual_tolerance)
                      : solve_dense_linear_system(objFunc, x, grad, direction, current_residual_tolerance);

        if (std::isnan(residual) || residual > current_residual_tolerance)
        {
            m_logger.debug("[{}] large (or nan) linear solve residual {}>{} (‖∇f‖={})",
                           name(), residual, current_residual_tolerance, grad_norm);

            return false;
        }
        else
        {
            m_logger.trace("linear solve residual {}", residual);
        }

        return true;
    }

    // =======================================================================

    double Newton::solve_sparse_linear_system(Problem &objFunc,
                                              const TVector &x,
                                              const TVector &grad,
                                              TVector &direction,
                                              const double residual_target)
    {
        polysolve::StiffnessMatrix hessian;

        {
            POLYSOLVE_SCOPED_STOPWATCH("assembly time", this->assembly_time, m_logger);
            compute_hessian(objFunc, x, hessian);
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("linear solve", this->inverting_time, m_logger);

            const double grad_norm = objFunc.grad_norm(grad, norm_type);
            if (linear_solver->is_iterative() && grad_norm > 0 && std::isfinite(grad_norm))
                linear_solver->set_tolerance(residual_target / grad_norm);

            // TODO: get the correct size
            linear_solver->analyze_pattern(hessian, hessian.rows());

            try
            {
                linear_solver->factorize(hessian);
            }
            catch (const std::runtime_error &err)
            {
                // warn if using gradient descent
                m_logger.debug("Unable to factorize Hessian: \"{}\"", err.what());

                // Eigen::saveMarket(hessian, "problematic_hessian.mtx");
                return std::nan("");
            }

            linear_solver->solve(-grad, direction); // H Δx = -g
        }

        const TVector linear_model_residual = hessian * direction + grad; // H dx + g = 0
        const double residual = objFunc.grad_norm(linear_model_residual, norm_type);

        forcing_term_strategy->record_linear_model(grad, linear_model_residual);

        json info;
        linear_solver->get_info(info);
        internal_solver_info.push_back(info);
        log_linear_solver_info(m_logger, linear_solver->name(), info);

        return residual;
    }

    double Newton::solve_dense_linear_system(Problem &objFunc,
                                             const TVector &x,
                                             const TVector &grad,
                                             TVector &direction,
                                             const double residual_target)
    {
        Eigen::MatrixXd hessian;

        {
            POLYSOLVE_SCOPED_STOPWATCH("assembly time", this->assembly_time, m_logger);
            compute_hessian(objFunc, x, hessian);
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("linear solve", this->inverting_time, m_logger);

            const double grad_norm = objFunc.grad_norm(grad, norm_type);
            if (linear_solver->is_iterative() && grad_norm > 0 && std::isfinite(grad_norm))
                linear_solver->set_tolerance(residual_target / grad_norm);

            try
            {
                linear_solver->analyze_pattern_dense(hessian, hessian.rows());
                linear_solver->factorize_dense(hessian);
                linear_solver->solve(-grad, direction);
            }
            catch (const std::runtime_error &err)
            {
                // warn if using gradient descent
                m_logger.debug("Unable to factorize Hessian: \"{}\"",
                               err.what());

                return std::nan("");
            }
        }

        const TVector linear_model_residual = hessian * direction + grad; // H dx + g = 0
        const double residual = objFunc.grad_norm(linear_model_residual, norm_type);

        forcing_term_strategy->record_linear_model(grad, linear_model_residual);

        json info;
        linear_solver->get_info(info);
        internal_solver_info.push_back(info);
        log_linear_solver_info(m_logger, linear_solver->name(), info);

        return residual;
    }
    // =======================================================================

    void Newton::compute_hessian(Problem &objFunc,
                                 const TVector &x,
                                 polysolve::StiffnessMatrix &hessian)

    {
        objFunc.set_project_to_psd(false);
        objFunc.hessian(x, hessian);
    }

    void ProjectedNewton::compute_hessian(Problem &objFunc,
                                          const TVector &x,
                                          polysolve::StiffnessMatrix &hessian)

    {
        objFunc.set_project_to_psd(true);
        objFunc.hessian(x, hessian);
    }

    void RegularizedNewton::compute_hessian(Problem &objFunc,
                                            const TVector &x,
                                            polysolve::StiffnessMatrix &hessian)

    {
        if (x.size() != x_cache.size() || x != x_cache)
        {
            objFunc.set_project_to_psd(project_to_psd);
            objFunc.hessian(x, hessian_cache);
            x_cache = x;
        }
        hessian = hessian_cache;
        if (reg_weight > 0)
        {
            hessian += reg_weight * sparse_identity(hessian.rows(), hessian.cols());
        }
    }

    void Newton::compute_hessian(Problem &objFunc,
                                 const TVector &x,
                                 Eigen::MatrixXd &hessian)

    {
        objFunc.set_project_to_psd(false);
        objFunc.hessian(x, hessian);
    }

    void ProjectedNewton::compute_hessian(Problem &objFunc,
                                          const TVector &x,
                                          Eigen::MatrixXd &hessian)

    {
        objFunc.set_project_to_psd(true);
        objFunc.hessian(x, hessian);
    }

    void RegularizedNewton::compute_hessian(Problem &objFunc,
                                            const TVector &x,
                                            Eigen::MatrixXd &hessian)

    {
        objFunc.set_project_to_psd(project_to_psd);
        objFunc.hessian(x, hessian);
        if (reg_weight > 0)
        {
            for (int k = 0; k < x.size(); k++)
                hessian(k, k) += reg_weight;
        }
    }
    // =======================================================================

    bool RegularizedNewton::handle_error()
    {
        reg_weight *= reg_weight_inc;
        return reg_weight < reg_weight_max;
    }
    // =======================================================================

    void Newton::update_solver_info(json &solver_info, const double per_iteration)
    {
        Superclass::update_solver_info(solver_info, per_iteration);

        solver_info["internal_solver"] = internal_solver_info;
        solver_info["time_assembly"] = assembly_time / per_iteration;
        solver_info["time_inverting"] = inverting_time / per_iteration;
    }

    void Newton::reset_times()
    {
        assembly_time = 0;
        inverting_time = 0;
    }

    void Newton::log_times() const
    {
        if (assembly_time <= 0 && inverting_time <= 0)
            return; // nothing to log
        m_logger.debug(
            "[{}][{}] assembly: {:.2e}s; linear_solve: {:.2e}s",
            fmt::format(fmt::fg(fmt::terminal_color::magenta), "timing"),
            name(), assembly_time, inverting_time);
    }

    // =======================================================================

} // namespace polysolve::nonlinear
