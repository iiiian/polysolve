#include "Newton.hpp"

#include <polysolve/Utils.hpp>

#if defined(SPDLOG_FMT_EXTERNAL)
#include <fmt/color.h>
#else
#include <spdlog/fmt/bundled/color.h>
#endif

#include <GenEigsSolver.h>
#include <SymEigsSolver.h>
#include <MatOp/SparseGenMatProd.h>
#include <MatOp/DenseSymMatProd.h>
#include <MatOp/SparseSymMatProd.h>
#include <fstream>

namespace polysolve::nonlinear
{

    std::vector<std::shared_ptr<DescentStrategy>> Newton::create_solver(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger)
    {
        // Copies stuff from main newton
        json proj_solver_params = R"({"ProjectedNewton": {}})"_json;
        proj_solver_params["ProjectedNewton"]["residual_tolerance"] = solver_params["Newton"]["residual_tolerance"];
        proj_solver_params["ProjectedNewton"]["compare_to_full"] = solver_params["Newton"]["compare_to_full"];

        json ppn_solver_params = R"({"ProgressivelyProjectedNewton": {}})"_json;
        ppn_solver_params["ProgressivelyProjectedNewton"]["residual_tolerance"] = solver_params["Newton"]["residual_tolerance"];
        ppn_solver_params["ProgressivelyProjectedNewton"]["compare_to_full"] = solver_params["Newton"]["compare_to_full"];
        ppn_solver_params["ProgressivelyProjectedNewton"]["alpha"] = solver_params["Newton"]["alpha"];
        ppn_solver_params["ProgressivelyProjectedNewton"]["beta"] = solver_params["Newton"]["beta"];
        ppn_solver_params["ProgressivelyProjectedNewton"]["max_attempts"] = solver_params["Newton"]["max_attempts"];

        json reg_solver_params = R"({"RegularizedNewton": {}})"_json;
        reg_solver_params["RegularizedNewton"]["residual_tolerance"] = solver_params["Newton"]["residual_tolerance"];
        reg_solver_params["RegularizedNewton"]["reg_weight_min"] = solver_params["Newton"]["reg_weight_min"];
        reg_solver_params["RegularizedNewton"]["reg_weight_max"] = solver_params["Newton"]["reg_weight_max"];
        reg_solver_params["RegularizedNewton"]["reg_weight_inc"] = solver_params["Newton"]["reg_weight_inc"];
        reg_solver_params["RegularizedNewton"]["compare_to_full"] = solver_params["Newton"]["compare_to_full"];

        std::vector<std::shared_ptr<DescentStrategy>> res;

        const bool force_ppn = solver_params["Newton"]["force_ppn"];
        if (force_ppn) {
            res.push_back(std::make_unique<ProgressivelyProjectedNewton>(
                sparse, ppn_solver_params, linear_solver_params,
                characteristic_length, logger));

            /*const double reg_weight_min = solver_params["Newton"]["reg_weight_min"];
            if (reg_weight_min > 0)
                res.push_back(std::make_unique<RegularizedNewton>(
                    sparse, solver_params["Newton"]["use_psd_projection_in_regularized"],
                    reg_solver_params, linear_solver_params,
                    characteristic_length, logger));*/

            return res;
        }

        const bool force_psd_projection = solver_params["Newton"]["force_psd_projection"];
        if (!force_psd_projection)
            res.push_back(std::make_unique<Newton>(
                sparse,
                solver_params, linear_solver_params,
                characteristic_length, logger));

        const bool use_psd_projection = solver_params["Newton"]["use_psd_projection"];
        if (use_psd_projection)
            res.push_back(std::make_unique<ProjectedNewton>(
                sparse,
                proj_solver_params, linear_solver_params,
                characteristic_length, logger));

        const double reg_weight_min = solver_params["Newton"]["reg_weight_min"];
        if (reg_weight_min > 0)
            res.push_back(std::make_unique<RegularizedNewton>(
                sparse, solver_params["Newton"]["use_psd_projection_in_regularized"],
                reg_solver_params, linear_solver_params,
                characteristic_length, logger));

        if (res.empty())
            log_and_throw_error(logger, "Newton needs to have at least one of force_psd_projection=false, reg_weight_min>0, or use_psd_projection=true");

        return res;
    }

    Newton::Newton(const bool sparse,
                   const double residual_tolerance,
                   const json &solver_params,
                   const json &linear_solver_params,
                   const double characteristic_length,
                   const bool try_neg_eig_dir,
                   spdlog::logger &logger)
        : Superclass(solver_params, characteristic_length, logger),
          is_sparse(sparse), characteristic_length(characteristic_length), residual_tolerance(residual_tolerance), try_neg_eig_dir(try_neg_eig_dir)
    {
        linear_solver = polysolve::linear::Solver::create(linear_solver_params, logger);
        if (linear_solver->is_dense() == sparse)
            log_and_throw_error(logger, "Newton linear solver must be {}, instead got {}", sparse ? "sparse" : "dense", linear_solver->name());

        if (residual_tolerance <= 0)
            log_and_throw_error(logger, "Newton residual_tolerance must be > 0, instead got {}", residual_tolerance);
    }

    Newton::Newton(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger)
        : Newton(sparse, extract_param("Newton", "residual_tolerance", solver_params), solver_params, linear_solver_params, characteristic_length, solver_params["Newton"]["try_neg_eig_dir"], logger)
    {
    }

    ProjectedNewton::ProjectedNewton(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger)
        : Superclass(sparse, extract_param("ProjectedNewton", "residual_tolerance", solver_params), solver_params, linear_solver_params, characteristic_length, false, logger), compare_to_full(solver_params["ProjectedNewton"]["compare_to_full"])
    {
    }

    ProgressivelyProjectedNewton::ProgressivelyProjectedNewton(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger)
        : Superclass(sparse, extract_param("ProgressivelyProjectedNewton", "residual_tolerance", solver_params), solver_params, linear_solver_params, characteristic_length, false, logger), compare_to_full(solver_params["ProgressivelyProjectedNewton"]["compare_to_full"]), alpha(solver_params["ProgressivelyProjectedNewton"]["alpha"]), beta(solver_params["ProgressivelyProjectedNewton"]["beta"]), max_attempts(solver_params["ProgressivelyProjectedNewton"]["max_attempts"])
    {
    }

    RegularizedNewton::RegularizedNewton(
        const bool sparse,
        const bool project_to_psd,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger)
        : Superclass(sparse, extract_param("RegularizedNewton", "residual_tolerance", solver_params), solver_params, linear_solver_params, characteristic_length, false, logger),
          project_to_psd(project_to_psd), compare_to_full(solver_params["RegularizedNewton"]["compare_to_full"])
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

    void Newton::reset(const int ndof)
    {
        Superclass::reset(ndof);
        internal_solver_info = json::array();
    }

    void RegularizedNewton::reset(const int ndof)
    {
        Superclass::reset(ndof);
        reg_weight = reg_weight_min;
    }

    void ProgressivelyProjectedNewton::reset(const int ndof)
    {
        Superclass::reset(ndof);
        double d = std::numeric_limits<double>::infinity();
        last_residual.resize(0);
        curr_attempts = 0;
    }

    // =======================================================================

    bool Newton::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        const double residual =
            is_sparse ? solve_sparse_linear_system(objFunc, x, grad, direction)
                      : solve_dense_linear_system(objFunc, x, grad, direction);

        if (std::isnan(residual) || residual > residual_tolerance * characteristic_length)
        {
            m_logger.debug("[{}] large (or nan) linear solve residual {}>{} (‖∇f‖={})",
                           name(), residual, residual_tolerance * characteristic_length, grad.norm());

            if (!try_neg_eig_dir)
            {
                return false;
            }

            polysolve::StiffnessMatrix hessian;
            compute_hessian(objFunc, x, hessian);
            Spectra::SparseSymMatProd<double> op(hessian);
            Spectra::SymEigsSolver<double, Spectra::SMALLEST_ALGE, Spectra::SparseSymMatProd<double>> eigs(&op, 1, 6);

            eigs.init();
            int nconv = eigs.compute();
            Eigen::VectorXd eigenvalues;
            Eigen::MatrixXd eigenvectors;
            if (eigs.info() == Spectra::SUCCESSFUL)
            {
                eigenvalues = eigs.eigenvalues();
                eigenvectors = eigs.eigenvectors();

                m_logger.debug("eigenvalue found: {}", eigenvalues(0)); 
                direction = eigenvectors.col(0);

                return true;
            }

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
                                              TVector &direction)
    {
        polysolve::StiffnessMatrix hessian;

        {
            POLYSOLVE_SCOPED_STOPWATCH("assembly time", this->assembly_time, m_logger);
            compute_hessian(objFunc, x, hessian);
        }

        std::set<int> bad_indices;
        {
            POLYSOLVE_SCOPED_STOPWATCH("bad dof time", this->bad_dof_time, m_logger);
            linear_solver->set_positions(positions);
            linear_solver->set_elements(elements);
            linear_solver->set_dof_to_function(dof_to_function);
            linear_solver->set_problematic_dofs(problematic_indices);
            linear_solver->reduced_to_full_func = objFunc.reduced_to_full_func;
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("linear solve", this->inverting_time, m_logger);
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

        last_residual = hessian * direction + grad;
        const double residual = last_residual.norm(); // H Δx + g = 0

        json info;
        linear_solver->get_info(info);
        internal_solver_info.push_back(info);

        return residual;
    }

    double Newton::solve_dense_linear_system(Problem &objFunc,
                                             const TVector &x,
                                             const TVector &grad,
                                             TVector &direction)
    {
        Eigen::MatrixXd hessian;

        {
            POLYSOLVE_SCOPED_STOPWATCH("assembly time", this->assembly_time, m_logger);
            compute_hessian(objFunc, x, hessian);
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("linear solve", this->inverting_time, m_logger);

            try
            {
                linear_solver->set_positions(positions);
                linear_solver->set_elements(elements);
                linear_solver->set_problematic_dofs(problematic_indices);
                linear_solver->set_dof_to_function(dof_to_function);
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

        const double residual = (hessian * direction + grad).norm(); // H Δx + g = 0

        json info;
        linear_solver->get_info(info);
        internal_solver_info.push_back(info);

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

        if (compare_to_full)
        {
            polysolve::StiffnessMatrix full_hessian;
            objFunc.set_project_to_psd(false);
            objFunc.hessian(x, full_hessian);

            polysolve::StiffnessMatrix diff_hessian = full_hessian - hessian;
            Eigen::MatrixXd HTH = diff_hessian.transpose() * diff_hessian;

            Spectra::DenseSymMatProd<double> op(HTH);
            Spectra::SymEigsSolver<double, Spectra::LARGEST_MAGN, Spectra::DenseSymMatProd<double>> eigs(&op, 1, 6);

            eigs.init();
            int nconv = eigs.compute();
            Eigen::VectorXd eigenvalues;
            if (eigs.info() == Spectra::SUCCESSFUL)
                eigenvalues = eigs.eigenvalues();

            double largestSingularValue = eigenvalues(0); 

            m_logger.trace("L2 Norm of Hessian - Proj(Hessian): {}", largestSingularValue);

            Eigen::SimplicialLDLT<polysolve::StiffnessMatrix> chol_decomp(full_hessian);
            bool spd = !(chol_decomp.info() == Eigen::NumericalIssue);
            m_logger.trace("Hessian isSPD: {}", spd);

        }
    }

    void ProgressivelyProjectedNewton::compute_hessian(Problem &objFunc,
                                          const TVector &x,
                                          polysolve::StiffnessMatrix &hessian)
    {
        objFunc.set_project_to_psd(false);
        if (std::isinf(d))
        {
            objFunc.projection_setting = 0;
        }
        else {
            objFunc.projection_setting = 3;
            objFunc.dofs_to_project.clear();
            for (int i = 0; i < last_residual.size(); ++i)
            {
                if (std::abs(last_residual(i)) > d) {
                    objFunc.dofs_to_project.insert(i);
                }
            }
        }
        objFunc.hessian(x, hessian);

        if (compare_to_full)
        {
            polysolve::StiffnessMatrix full_hessian;
            objFunc.set_project_to_psd(false);
            objFunc.projection_setting = 0;
            objFunc.hessian(x, full_hessian);

            polysolve::StiffnessMatrix diff_hessian = full_hessian - hessian;
            Eigen::MatrixXd HTH = diff_hessian.transpose() * diff_hessian;

            Spectra::DenseSymMatProd<double> op(HTH);
            Spectra::SymEigsSolver<double, Spectra::LARGEST_MAGN, Spectra::DenseSymMatProd<double>> eigs(&op, 1, 6);

            eigs.init();
            int nconv = eigs.compute();
            Eigen::VectorXd eigenvalues;
            if (eigs.info() == Spectra::SUCCESSFUL)
                eigenvalues = eigs.eigenvalues();

            double largestSingularValue = eigenvalues(0); 

            m_logger.trace("L2 Norm of Hessian - Proj(Hessian): {}", largestSingularValue);

            Eigen::SimplicialLDLT<polysolve::StiffnessMatrix> chol_decomp(full_hessian);
            bool spd = !(chol_decomp.info() == Eigen::NumericalIssue);
            m_logger.trace("Hessian isSPD: {}", spd);

        }
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

        if (compare_to_full)
        {
            polysolve::StiffnessMatrix full_hessian;
            objFunc.set_project_to_psd(false);
            objFunc.hessian(x, full_hessian);

            polysolve::StiffnessMatrix diff_hessian = full_hessian - hessian;
            Eigen::MatrixXd HTH = diff_hessian.transpose() * diff_hessian;

            Spectra::DenseSymMatProd<double> op(HTH);
            Spectra::SymEigsSolver<double, Spectra::LARGEST_MAGN, Spectra::DenseSymMatProd<double>> eigs(&op, 1, 6);

            eigs.init();
            int nconv = eigs.compute();
            Eigen::VectorXd eigenvalues;
            if (eigs.info() == Spectra::SUCCESSFUL)
                eigenvalues = eigs.eigenvalues();

            double largestSingularValue = eigenvalues(0); 

            m_logger.trace("L2 Norm of Hessian - Proj(Hessian): {}", largestSingularValue);

            Eigen::SimplicialLDLT<polysolve::StiffnessMatrix> chol_decomp(full_hessian);
            bool spd = !(chol_decomp.info() == Eigen::NumericalIssue);
            m_logger.trace("Hessian isSPD: {}", spd);

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

        if (compare_to_full)
        {
            Eigen::MatrixXd full_hessian;
            objFunc.set_project_to_psd(false);
            objFunc.hessian(x, full_hessian);

            Eigen::MatrixXd diff_hessian = full_hessian - hessian;
            Eigen::MatrixXd HTH = diff_hessian.transpose() * diff_hessian;

            Spectra::DenseSymMatProd<double> op(HTH);
            Spectra::SymEigsSolver<double, Spectra::LARGEST_MAGN, Spectra::DenseSymMatProd<double>> eigs(&op, 1, 6);

            eigs.init();
            int nconv = eigs.compute();
            Eigen::VectorXd eigenvalues;
            if (eigs.info() == Spectra::SUCCESSFUL)
                eigenvalues = eigs.eigenvalues();

            double largestSingularValue = eigenvalues(0); 

            m_logger.trace("L2 Norm of Hessian - Proj(Hessian): {}", largestSingularValue);
        }
    }

    void ProgressivelyProjectedNewton::compute_hessian(Problem &objFunc,
                                          const TVector &x,
                                          Eigen::MatrixXd &hessian)
    {
        objFunc.projection_setting = 3;
        objFunc.hessian(x, hessian);

        if (compare_to_full)
        {
            Eigen::MatrixXd full_hessian;
            objFunc.set_project_to_psd(false);
            objFunc.projection_setting = 0;
            objFunc.hessian(x, full_hessian);

            Eigen::MatrixXd diff_hessian = full_hessian - hessian;
            Eigen::MatrixXd HTH = diff_hessian.transpose() * diff_hessian;

            Spectra::DenseSymMatProd<double> op(HTH);
            Spectra::SymEigsSolver<double, Spectra::LARGEST_MAGN, Spectra::DenseSymMatProd<double>> eigs(&op, 1, 6);

            eigs.init();
            int nconv = eigs.compute();
            Eigen::VectorXd eigenvalues;
            if (eigs.info() == Spectra::SUCCESSFUL)
                eigenvalues = eigs.eigenvalues();

            double largestSingularValue = eigenvalues(0); 

            m_logger.trace("L2 Norm of Hessian - Proj(Hessian): {}", largestSingularValue);
        }
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

    void RegularizedNewton::handle_success()
    {
        if (reg_weight > reg_weight_min)
        {
            reg_weight /= reg_weight_inc;
        }
    }

    bool ProgressivelyProjectedNewton::handle_error()
    {
        if (std::isinf(d))
        {
            d = alpha * last_residual.cwiseAbs().maxCoeff();
        }
        else 
        {
            d *= alpha;
        }

        ++curr_attempts;
        return curr_attempts < max_attempts && d > 0;
    }

    void ProgressivelyProjectedNewton::handle_success()
    {
        d *= beta;
        if (curr_attempts > 0)
        {
            --curr_attempts;
        }

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
