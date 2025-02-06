#pragma once

#include <polysolve/Utils.hpp>

#include <polysolve/nonlinear/Problem.hpp>

namespace polysolve::nonlinear
{

    class DescentStrategy
    {
    public:
        using TVector = Problem::TVector;
        using Scalar = Problem::Scalar;

        /// @brief Constructor 
        /// @param solver_params_ JSON of solver parameters
        /// @param characteristic_length 
        /// @param logger 
        DescentStrategy(const json &solver_params_,
                        const double characteristic_length,
                        spdlog::logger &logger)
            : m_logger(logger)
        {
        }
        virtual ~DescentStrategy() {}
        virtual std::string name() const = 0;

        virtual void reset(const int ndof) {}
        virtual void reset_times() {}

        /// @brief Update solver info after finding descent direction
        /// @param solver_info JSON of solver parameters
        /// @param per_iteration Number of iterations (used to normalize timings)
        virtual void update_solver_info(json &solver_info, const double per_iteration) {}
        virtual void log_times() const {}

        virtual bool is_direction_descent() { return true; }
        virtual bool handle_error() { return false; }

        /// @brief Compute descent direction along which to do line search
        /// @param objFunc Problem to be minimized
        /// @param x Current input (n x 1)
        /// @param grad Current gradient (n x 1)
        /// @param direction Current descent direction (n x 1)
        /// @return True if a descent direction was successfully found
        virtual bool compute_update_direction(
            Problem &objFunc,
            const TVector &x,
            const TVector &grad,
            TVector &direction) = 0;

        void set_positions(Eigen::MatrixXd &new_positions) {positions = new_positions;}
        void set_elements(Eigen::MatrixXi &new_elements) {elements = new_elements;}
        void set_problematic_indices(std::vector<std::set<int>> &problematic_indices_) 
        {
            problematic_indices = problematic_indices_;
        }
        void set_dof_to_func_mapping(std::vector<int> &dof_to_function_) 
        {
            dof_to_function = dof_to_function_;
        }

    protected:
        spdlog::logger &m_logger;
        Eigen::MatrixXd positions;
        Eigen::MatrixXi elements;
        std::vector<std::set<int>> problematic_indices;
        std::vector<int> dof_to_function;
    };
} // namespace polysolve::nonlinear