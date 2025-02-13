#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"
#include "../Utils.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

#include <HYPRE_utilities.h>
#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>

#ifdef POLYSOLVE_WITH_ICHOL
#include <boost/property_tree/ptree.hpp>
namespace mschol {
    class ichol_precond;
}
#endif

namespace polysolve::linear
{

    class ExperimentalSolver : public Solver
    {

    public:
        ExperimentalSolver();
        ~ExperimentalSolver();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(ExperimentalSolver)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void set_parameters(const json &params) override;

        // Retrieve memory information from Pardiso
        virtual void get_info(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyze_pattern(const StiffnessMatrix &A, const int precond_num) override { precond_num_ = precond_num; }

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "Experimental"; }

    protected:
        int dimension_ = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)
        int max_iter_ = 1000;
        int pre_max_iter_ = 1;
        double conv_tol_ = 1e-10;

        // solver tuning options
        double theta = 0.5;
        bool nodal_coarsening = false;
        bool interp_rbms = false;
        bool do_mixed_precond = false;
        bool dss_in_middle = true;
        bool print_conditioning = false;
        bool use_incomplete_cholesky_precond = false;
        bool use_absolute_tol = false;
        bool select_bad_dofs_from_rhs;

        HYPRE_Int num_iterations;
        HYPRE_Complex final_res_norm;

    private:
        bool has_matrix_ = false;
        int precond_num_;

        Eigen::MatrixXd eigen_A;
        Eigen::LDLT<Eigen::MatrixXd> D_solver;

        #ifdef POLYSOLVE_WITH_ICHOL
        std::shared_ptr<mschol::ichol_precond> inc_chol_precond; // just to show it compiles
        boost::property_tree::ptree pt;
        Eigen::VectorXi ichol_dof_remapping;
        double rho = 1.01;
        #endif

        HYPRE_IJMatrix A;
        HYPRE_ParCSRMatrix parcsr_A;

        double copy_b_and_x_time;
        double set_options_time;
        double actual_solve_time;

        void custom_mixed_precond_iter(const HYPRE_Solver &precond, const Eigen::VectorXd &r, Eigen::VectorXd &z);
        void amg_precond_iter(const HYPRE_Solver &precond, const Ref<const VectorXd> b, Eigen::VectorXd &x);
        void dss_precond_iter(const Eigen::VectorXd &z, const Eigen::VectorXd &r, Eigen::VectorXd &next_z);
    };

} // namespace polysolve::linear
