#pragma once

#include "Solver.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#ifdef POLYSOLVE_WITH_MKL
#include <Eigen/PardisoSupport>
#endif

#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>
#include <HYPRE_utilities.h>

#include <spdlog/spdlog.h>

#include <deque>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

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
        void set_parameters(const json &params) override;
        void get_info(json &params) const override;

        void analyze_pattern(const StiffnessMatrix &A, const int precond_num) override;
        void factorize(const StiffnessMatrix &A) override;
        void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        std::string name() const override { return "Experimental"; }
        void set_tolerance(const double tol) override { conv_tol_ = tol; }

        void set_logger(spdlog::logger *logger) { logger_ = logger; }

    private:
        struct DisjointSet
        {
            explicit DisjointSet(const int n);
            int find_set(const int v);
            void union_set(const int a, const int b);

            std::vector<int> parent;
            std::vector<int> size;
        };

        class AbstractSolver
        {
        public:
            virtual void compute(const Eigen::SparseMatrix<double> &A) = 0;
            virtual Eigen::VectorXd solve(const Eigen::VectorXd &b) = 0;
            virtual ~AbstractSolver() = default;
        };

        template <typename EigenSolverT>
        class EigenWrapper final : public AbstractSolver
        {
        public:
            void compute(const Eigen::SparseMatrix<double> &A) override { solver_.compute(A); }
            Eigen::VectorXd solve(const Eigen::VectorXd &b) override { return solver_.solve(b); }

        private:
            EigenSolverT solver_;
        };

    private:
        spdlog::logger &logger();

        void eigen_to_hypre_par_vec(HYPRE_ParVector &par_x, HYPRE_IJVector &ij_x, const Eigen::VectorXd &x) const;
        void hypre_vec_to_eigen(const HYPRE_IJVector &ij_x, Eigen::Ref<Eigen::VectorXd> x) const;

        void HypreBoomerAMG_SetDefaultOptions(HYPRE_Solver &amg_precond) const;
        void HypreBoomerAMG_SetElasticityOptions(HYPRE_Solver &amg_precond) const;

        void matmul(const Eigen::VectorXd &x, Eigen::VectorXd &result) const;

        void prepare_dss(const Eigen::VectorXd &rhs, const HYPRE_Solver &amg_precond);
        void select_bad_indices(const Eigen::VectorXd &rhs, const HYPRE_Solver &amg_precond);
        void factorize_submatrices();

        void amg_precond_iter(const HYPRE_Solver &amg_precond, const Eigen::VectorXd &b, Eigen::VectorXd &x) const;
        void dss_precond_iter(const Eigen::VectorXd &z, const Eigen::VectorXd &r, Eigen::VectorXd &next_z) const;
        void custom_mixed_precond_iter(const HYPRE_Solver &amg_precond, const Eigen::VectorXd &r, Eigen::VectorXd &z) const;

        void pcg_solve(const Ref<const VectorXd> rhs, Ref<VectorXd> result, const HYPRE_Solver &amg_precond);

    private:
        spdlog::logger *logger_ = nullptr;

        int dimension_ = 1;
        int max_iter_ = 1000;
        int pre_max_iter_ = 1;
        double conv_tol_ = 1e-10;

        // solver tuning options
        double theta_ = 0.5;
        bool nodal_coarsening_ = false;
        bool interp_rbms_ = false; // not supported without mesh data (CLI)
        bool do_mixed_precond_ = false;
        bool dss_in_middle_ = true;
        bool print_conditioning_ = false; // retained for parity; no-op in CLI path
        bool use_absolute_tol_ = false;
        bool save_selection_criteria_ = false;
        bool save_problem_ = false;
        bool save_selected_indices_ = false;
        bool select_bad_dofs_from_rhs_ = false;
        bool select_bad_dofs_from_row_norms_ = false;
        bool select_bad_dofs_from_amg_ = false;
        double bad_dof_threshold_ = 0.1;
        int project_d_option_ = 0;
        bool jacobi_precond_ = false;
        bool decompose_subdomains_ = false;

        HYPRE_Int num_iterations_ = 0;
        HYPRE_Complex final_res_norm_ = 0;

        std::vector<double> iteration_residual_norm_;
        std::vector<double> iteration_time_s_;

        bool has_matrix_ = false;
        int precond_num_ = 0;

        Eigen::SparseMatrix<double, Eigen::RowMajor> A_row_;
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> diag_inv_;

        std::vector<std::set<int>> bad_indices_;
        std::vector<std::vector<int>> bad_indices_arrays_;
        std::vector<std::unordered_map<int, int>> index_mappings_;

        std::deque<std::unique_ptr<AbstractSolver>> D_solvers_;

        HYPRE_IJMatrix A_ij_;
        HYPRE_ParCSRMatrix A_par_;

        std::vector<HYPRE_BigInt> hypre_indices_;
    };

} // namespace polysolve::linear
