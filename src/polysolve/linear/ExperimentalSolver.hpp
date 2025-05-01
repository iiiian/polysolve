#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"
#include "../Utils.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include <deque>
#include <Eigen/SparseCholesky>

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
        double bad_dof_grad_threshold;
        bool save_grad_norms;
        bool save_selected_indices;
        bool save_problem;

        HYPRE_Int num_iterations;
        HYPRE_Complex final_res_norm;

        int num_threads = 1; 

    private:
        bool has_matrix_ = false;
        int precond_num_;

        int myid = 0;
        int num_procs = 1;
        int start_i, end_i;

        Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_A;
        std::deque<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>> D_solvers;

#ifdef POLYSOLVE_WITH_ICHOL
        std::shared_ptr<mschol::ichol_precond> inc_chol_precond; 
        boost::property_tree::ptree pt;
        Eigen::VectorXi ichol_dof_remapping;
        double rho = 1.01;
        int remap_dof(const int index);
#endif

        HYPRE_IJMatrix A;
        HYPRE_ParCSRMatrix parcsr_A;

        double copy_b_and_x_time;
        double set_options_time;
        double actual_solve_time;
        double factorization_time;
        double bad_dof_selection_time;
        double dss_factorization_time;
        double dss_assembly_time;

        void custom_mixed_precond_iter(const HYPRE_Solver &precond, const Eigen::VectorXd &r, Eigen::VectorXd &z);
        void amg_precond_iter(const HYPRE_Solver &precond, const Ref<const VectorXd> b, Eigen::VectorXd &x);
        void dss_precond_iter(const Eigen::VectorXd &z, const Eigen::VectorXd &r, Eigen::VectorXd &next_z);
        void matmul(Eigen::VectorXd &x, Eigen::SparseMatrix<double, Eigen::RowMajor> &A, Eigen::VectorXd &result);

        void check_matrix_conditioning(const std::string name, const std::set<int>& subdomain);
        void check_matrix_conditioning(const std::string name, const Eigen::MatrixXd& mat);

        void factorize_submatrix();
    };

} // namespace polysolve::linear
