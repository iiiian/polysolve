#ifdef POLYSOLVE_WITH_HYPRE

#include "ExperimentalSolver.hpp"

#include <polysolve/Utils.hpp>

#include <HYPRE_krylov.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>

#include <omp.h>

#ifndef HYPRE_SEQUENTIAL
#include <mpi.h>
#endif

namespace polysolve::linear
{
    namespace
    {
        using Clock = std::chrono::steady_clock;
        double seconds_since(const Clock::time_point &start)
        {
            return std::chrono::duration<double>(Clock::now() - start).count();
        }
    } // namespace

    // -----------------------------------------------------------------------------

    ExperimentalSolver::DisjointSet::DisjointSet(const int n)
    {
        parent.resize(n);
        size.assign(n, 1);
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }

    int ExperimentalSolver::DisjointSet::find_set(const int v)
    {
        if (v == parent[v])
            return v;
        parent[v] = find_set(parent[v]);
        return parent[v];
    }

    void ExperimentalSolver::DisjointSet::union_set(const int a, const int b)
    {
        int ra = find_set(a);
        int rb = find_set(b);
        if (ra == rb)
            return;
        if (size[ra] < size[rb])
            std::swap(ra, rb);
        parent[rb] = ra;
        size[ra] += size[rb];
    }

    // -----------------------------------------------------------------------------

    spdlog::logger &ExperimentalSolver::logger()
    {
        if (logger_ != nullptr)
            return *logger_;
        auto *def = spdlog::default_logger_raw();
        if (def != nullptr)
            return *def;
        throw std::runtime_error("ExperimentalSolver requires a logger (set_logger)");
    }

    ExperimentalSolver::ExperimentalSolver()
    {
#ifndef HYPRE_SEQUENTIAL
        int done_already = 0;
        MPI_Initialized(&done_already);
        if (!done_already)
        {
            MPI_Init(nullptr, nullptr);
        }
#endif

        const char *num_threads_val = std::getenv("OMP_NUM_THREADS");
        if (num_threads_val != nullptr)
        {
            const int num_threads = std::max(1, std::atoi(num_threads_val));
            Eigen::setNbThreads(num_threads);
        }
    }

    ExperimentalSolver::~ExperimentalSolver()
    {
        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A_ij_);
            has_matrix_ = false;
        }
    }

    // -----------------------------------------------------------------------------

    void ExperimentalSolver::set_parameters(const json &params)
    {
        if (!params.contains("Experimental"))
            return;

        const auto &p = params["Experimental"];

        if (p.contains("max_iter"))
            max_iter_ = p["max_iter"];
        if (p.contains("pre_max_iter"))
            pre_max_iter_ = p["pre_max_iter"];
        if (p.contains("tolerance"))
            conv_tol_ = p["tolerance"];

        if (p.contains("theta"))
            theta_ = p["theta"];
        if (p.contains("nodal_coarsening"))
            nodal_coarsening_ = p["nodal_coarsening"];
        if (p.contains("interp_rbms"))
            interp_rbms_ = p["interp_rbms"];
        if (p.contains("dimension"))
            dimension_ = p["dimension"];

        if (p.contains("do_mixed_precond"))
            do_mixed_precond_ = p["do_mixed_precond"];
        if (p.contains("print_conditioning"))
            print_conditioning_ = p["print_conditioning"];
        if (p.contains("save_problem"))
            save_problem_ = p["save_problem"];
        if (p.contains("save_selection_criteria"))
            save_selection_criteria_ = p["save_selection_criteria"];
        if (p.contains("save_selected_indices"))
            save_selected_indices_ = p["save_selected_indices"];
        if (p.contains("dss_in_middle"))
            dss_in_middle_ = p["dss_in_middle"];
        if (p.contains("use_absolute_tol"))
            use_absolute_tol_ = p["use_absolute_tol"];

        if (p.contains("select_bad_dofs_from_row_norms"))
            select_bad_dofs_from_row_norms_ = p["select_bad_dofs_from_row_norms"];
        if (p.contains("select_bad_dofs_from_amg"))
            select_bad_dofs_from_amg_ = p["select_bad_dofs_from_amg"];
        if (p.contains("select_bad_dofs_from_rhs"))
            select_bad_dofs_from_rhs_ = p["select_bad_dofs_from_rhs"];
        if (p.contains("bad_dof_threshold"))
            bad_dof_threshold_ = p["bad_dof_threshold"];

        if (p.contains("project_d_option"))
            project_d_option_ = p["project_d_option"];
        if (p.contains("jacobi_precond"))
            jacobi_precond_ = p["jacobi_precond"];
        if (p.contains("decompose_subdomains"))
            decompose_subdomains_ = p["decompose_subdomains"];

        // PCG only (for this port)
        if (p.contains("use_gmres") && p["use_gmres"].get<bool>())
            throw std::runtime_error("ExperimentalSolver: GMRES not supported in this port (PCG only)");
        if (p.contains("use_minres") && p["use_minres"].get<bool>())
            throw std::runtime_error("ExperimentalSolver: MINRES not supported in this port (PCG only)");

        if (interp_rbms_)
        {
            throw std::runtime_error("ExperimentalSolver: interp_rbms=true requires mesh positions; not supported in polysolve_cli");
        }
    }

    void ExperimentalSolver::get_info(json &params) const
    {
        params["solver_iter"] = num_iterations_;
        params["num_iterations"] = num_iterations_;
        params["final_res_norm"] = final_res_norm_;

        if (!iteration_residual_norm_.empty())
            params["iteration_residual_norm"] = iteration_residual_norm_;
        if (!iteration_time_s_.empty())
            params["iteration_time_s"] = iteration_time_s_;
    }

    // -----------------------------------------------------------------------------

    void ExperimentalSolver::analyze_pattern(const StiffnessMatrix &A, const int precond_num)
    {
        precond_num_ = precond_num;

        double eigen_copy_time = 0.0;
        {
            POLYSOLVE_SCOPED_STOPWATCH("experimental eigen matrix copy time", eigen_copy_time, logger());
            A_row_ = A;
        }
    }

    void ExperimentalSolver::factorize(const StiffnessMatrix &A)
    {
        (void)A;
        if (precond_num_ <= 0)
        {
            throw std::runtime_error("ExperimentalSolver::analyze_pattern must be called before factorize");
        }

        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A_ij_);
            has_matrix_ = false;
        }

        if (jacobi_precond_)
        {
            diag_inv_ = A_row_.diagonal().asDiagonal().inverse();
        }

        if (save_problem_)
        {
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(static_cast<size_t>(A_row_.nonZeros()));
            for (int k = 0; k < A_row_.outerSize(); ++k)
            {
                for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A_row_, k); it; ++it)
                {
                    triplets.emplace_back(it.row(), it.col(), it.value());
                }
            }

            std::ofstream file("A.mat", std::ios_base::app);
            file << A_row_.rows() << " " << A_row_.cols() << " " << A_row_.nonZeros() << "\n";
            for (const auto &t : triplets)
            {
                file << t.row() << " " << t.col() << " " << t.value() << " ";
            }
            file << "\n";
        }

        has_matrix_ = true;

        const HYPRE_Int rows = static_cast<HYPRE_Int>(A_row_.rows());
        const HYPRE_Int cols = static_cast<HYPRE_Int>(A_row_.cols());

        hypre_indices_.resize(static_cast<size_t>(rows));
        for (HYPRE_Int i = 0; i < rows; ++i)
            hypre_indices_[static_cast<size_t>(i)] = i;

#ifndef HYPRE_SEQUENTIAL
        HYPRE_IJMatrixCreate(
            MPI_COMM_WORLD,
            static_cast<HYPRE_BigInt>(0),
            static_cast<HYPRE_BigInt>(rows - 1),
            static_cast<HYPRE_BigInt>(0),
            static_cast<HYPRE_BigInt>(cols - 1),
            &A_ij_);
#else
        HYPRE_IJMatrixCreate(0, 0, rows - 1, 0, cols - 1, &A_ij_);
#endif
        HYPRE_IJMatrixSetObjectType(A_ij_, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A_ij_);

        for (HYPRE_Int r = 0; r < rows; ++r)
        {
            HYPRE_BigInt row[1] = {static_cast<HYPRE_BigInt>(r)};
            std::vector<HYPRE_BigInt> cols_i;
            std::vector<double> vals_i;
            cols_i.reserve(16);
            vals_i.reserve(16);

            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A_row_, r); it; ++it)
            {
                cols_i.push_back(static_cast<HYPRE_BigInt>(it.col()));
                vals_i.push_back(it.value());
            }

            HYPRE_Int ncols = static_cast<HYPRE_Int>(cols_i.size());
            if (ncols > 0)
            {
                HYPRE_IJMatrixSetValues(A_ij_, 1, &ncols, row, cols_i.data(), vals_i.data());
            }
        }

        HYPRE_IJMatrixAssemble(A_ij_);
        void *obj = nullptr;
        HYPRE_IJMatrixGetObject(A_ij_, &obj);
        A_par_ = static_cast<HYPRE_ParCSRMatrix>(obj);
    }

    // -----------------------------------------------------------------------------

    void ExperimentalSolver::eigen_to_hypre_par_vec(HYPRE_ParVector &par_x, HYPRE_IJVector &ij_x, const Eigen::VectorXd &x) const
    {
        const HYPRE_Int n = static_cast<HYPRE_Int>(x.size());
#ifndef HYPRE_SEQUENTIAL
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, static_cast<HYPRE_BigInt>(0), static_cast<HYPRE_BigInt>(n - 1), &ij_x);
#else
        HYPRE_IJVectorCreate(0, 0, n - 1, &ij_x);
#endif
        HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(ij_x);

        if (static_cast<size_t>(n) != hypre_indices_.size())
        {
            throw std::runtime_error("ExperimentalSolver internal error: hypre index buffer not initialized");
        }
        HYPRE_IJVectorSetValues(ij_x, n, hypre_indices_.data(), x.data());

        HYPRE_IJVectorAssemble(ij_x);
        HYPRE_IJVectorGetObject(ij_x, (void **)&par_x);
    }

    void ExperimentalSolver::hypre_vec_to_eigen(const HYPRE_IJVector &ij_x, Eigen::Ref<Eigen::VectorXd> x) const
    {
        const HYPRE_Int n = static_cast<HYPRE_Int>(x.size());
        if (static_cast<size_t>(n) != hypre_indices_.size())
        {
            throw std::runtime_error("ExperimentalSolver internal error: hypre index buffer not initialized");
        }
        HYPRE_IJVectorGetValues(ij_x, n, hypre_indices_.data(), x.data());
    }

    // -----------------------------------------------------------------------------

    void ExperimentalSolver::HypreBoomerAMG_SetDefaultOptions(HYPRE_Solver &amg_precond) const
    {
        // Coarsening
        const int coarsen_type = 10; // HMIS
        const int agg_levels = 1;

        // Interpolation
        const int interp_type = 6;
        const int Pmax = 4;

        // Relaxation
        const int relax_type = 8; // l1-GS
        const int relax_sweeps = 1;

        // Misc
        const int print_level = 0;
        const int max_levels = 25;

        HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
        HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
        HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);
        HYPRE_BoomerAMGSetNumSweeps(amg_precond, relax_sweeps);
        HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta_);
        HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type);
        HYPRE_BoomerAMGSetPMaxElmts(amg_precond, Pmax);
        HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level);
        HYPRE_BoomerAMGSetMaxLevels(amg_precond, max_levels);

        // Use as a preconditioner (one V-cycle, zero tolerance)
        HYPRE_BoomerAMGSetMaxIter(amg_precond, pre_max_iter_);
        HYPRE_BoomerAMGSetTol(amg_precond, 0.0);
    }

    void ExperimentalSolver::HypreBoomerAMG_SetElasticityOptions(HYPRE_Solver &amg_precond) const
    {
        HYPRE_BoomerAMGSetNumFunctions(amg_precond, dimension_);
        HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
        HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta_);

        if (nodal_coarsening_)
        {
            const int nodal = 4;
            const int nodal_diag = 1;
            const int relax_coarse = 8;
            HYPRE_BoomerAMGSetNodal(amg_precond, nodal);
            HYPRE_BoomerAMGSetNodalDiag(amg_precond, nodal_diag);
            HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_coarse, 3);
        }
    }

    // -----------------------------------------------------------------------------

    void ExperimentalSolver::matmul(const Eigen::VectorXd &x, Eigen::VectorXd &result) const
    {
        result = A_row_ * x;
    }

    // -----------------------------------------------------------------------------

    void ExperimentalSolver::amg_precond_iter(const HYPRE_Solver &amg_precond, const Eigen::VectorXd &eigen_b, Eigen::VectorXd &eigen_x) const
    {
        if (jacobi_precond_)
        {
            eigen_x = diag_inv_ * (eigen_b - (A_row_ * eigen_x - A_row_.diagonal().asDiagonal() * eigen_x));
            return;
        }

        HYPRE_ParVector par_x;
        HYPRE_IJVector x;
        HYPRE_ParVector par_b;
        HYPRE_IJVector b;

        eigen_to_hypre_par_vec(par_x, x, eigen_x);
        eigen_to_hypre_par_vec(par_b, b, eigen_b);

        HYPRE_BoomerAMGSolve(amg_precond, A_par_, par_b, par_x);

        hypre_vec_to_eigen(x, eigen_x);

        HYPRE_IJVectorDestroy(x);
        HYPRE_IJVectorDestroy(b);
    }

    void ExperimentalSolver::dss_precond_iter(const Eigen::VectorXd &z, const Eigen::VectorXd &r, Eigen::VectorXd &next_z) const
    {
        next_z = z;
        if (bad_indices_arrays_.empty())
            return;

        std::vector<Eigen::VectorXd> rhs_workspace;
        std::vector<Eigen::VectorXd> res_workspace;
        rhs_workspace.resize(static_cast<size_t>(std::max(1, omp_get_max_threads())));
        res_workspace.resize(static_cast<size_t>(std::max(1, omp_get_max_threads())));

#pragma omp parallel for
        for (int index = 0; index < static_cast<int>(bad_indices_arrays_.size()); ++index)
        {
            const int tid = omp_get_thread_num();
            const auto &subdomain = bad_indices_arrays_[index];

            Eigen::VectorXd &sub_rhs = rhs_workspace[static_cast<size_t>(tid)];
            Eigen::VectorXd &sub_result = res_workspace[static_cast<size_t>(tid)];

            sub_rhs.resize(static_cast<int>(subdomain.size()));
            sub_result.resize(static_cast<int>(subdomain.size()));

            for (int i = 0; i < static_cast<int>(subdomain.size()); ++i)
            {
                const int g = subdomain[static_cast<size_t>(i)];
                sub_rhs(index_mappings_[static_cast<size_t>(index)].at(g)) = r(g) - A_row_.row(g).dot(z);
            }

            sub_result = D_solvers_[static_cast<size_t>(index)]->solve(sub_rhs);

            for (int i = 0; i < static_cast<int>(subdomain.size()); ++i)
            {
                const int g = subdomain[static_cast<size_t>(i)];
                const int li = index_mappings_[static_cast<size_t>(index)].at(g);
#pragma omp atomic
                next_z(g) += sub_result(li);
            }
        }
    }

    void ExperimentalSolver::custom_mixed_precond_iter(const HYPRE_Solver &amg_precond, const Eigen::VectorXd &r, Eigen::VectorXd &z) const
    {
        if (!do_mixed_precond_ || bad_indices_arrays_.empty())
        {
            amg_precond_iter(amg_precond, r, z);
            return;
        }

        Eigen::VectorXd z1(r.size());
        Eigen::VectorXd z2(r.size());
        Eigen::VectorXd z3(r.size());
        z1.setZero();
        z2.setZero();
        z3.setZero();

        if (dss_in_middle_)
        {
            amg_precond_iter(amg_precond, r, z1);
            dss_precond_iter(z1, r, z2);
            Eigen::VectorXd A_times_z2;
            matmul(z2, A_times_z2);
            amg_precond_iter(amg_precond, r - A_times_z2, z3);
            z = z2 + z3;
        }
        else
        {
            Eigen::VectorXd z0(r.size());
            z0.setZero();
            dss_precond_iter(z0, r, z1);
            Eigen::VectorXd A_times_z1;
            matmul(z1, A_times_z1);
            amg_precond_iter(amg_precond, r - A_times_z1, z2);
            z2 += z1;
            dss_precond_iter(z2, r, z);
        }
    }

    // -----------------------------------------------------------------------------

    void ExperimentalSolver::select_bad_indices(const Eigen::VectorXd &rhs, const HYPRE_Solver &amg_precond)
    {
        bad_indices_.clear();
        bad_indices_.resize(1);

        if (!(select_bad_dofs_from_amg_ || select_bad_dofs_from_rhs_ || select_bad_dofs_from_row_norms_))
        {
            return;
        }
        if ((static_cast<int>(select_bad_dofs_from_amg_) + static_cast<int>(select_bad_dofs_from_rhs_)
             + static_cast<int>(select_bad_dofs_from_row_norms_))
            > 1)
        {
            logger().warn("Multiple selection methods specified; defaulting to row norms.");
            select_bad_dofs_from_amg_ = false;
            select_bad_dofs_from_rhs_ = false;
            select_bad_dofs_from_row_norms_ = true;
        }

        Eigen::VectorXd mags(rhs.size());
        mags.setZero();

        if (select_bad_dofs_from_rhs_)
        {
            if (dimension_ <= 1)
            {
                for (int i = 0; i < rhs.size(); ++i)
                    mags(i) = std::abs(rhs(i));
            }
            else
            {
                if (rhs.size() % dimension_ != 0)
                    throw std::runtime_error("ExperimentalSolver: rhs.size() must be divisible by dimension");
                for (int i = 0; i < rhs.size() / dimension_; ++i)
                {
                    double sq_mag = 0;
                    for (int j = 0; j < dimension_; ++j)
                    {
                        const double v = rhs(dimension_ * i + j);
                        sq_mag += v * v;
                    }
                    for (int j = 0; j < dimension_; ++j)
                    {
                        mags(dimension_ * i + j) = sq_mag;
                    }
                }
            }
        }

        if (select_bad_dofs_from_amg_)
        {
            HYPRE_Solver test_precond;
            HYPRE_BoomerAMGCreate(&test_precond);
            HypreBoomerAMG_SetDefaultOptions(test_precond);
            if (dimension_ > 1)
            {
                HypreBoomerAMG_SetElasticityOptions(test_precond);
            }
            HYPRE_BoomerAMGSetMaxIter(test_precond, 5);

            Eigen::VectorXd test_result = Eigen::VectorXd::Random(rhs.size());
            const Eigen::VectorXd start_result = test_result;
            Eigen::VectorXd test_rhs(rhs.size());
            test_rhs.setZero();

            HYPRE_ParVector par_b;
            HYPRE_IJVector b;
            HYPRE_ParVector par_x;
            HYPRE_IJVector x;
            eigen_to_hypre_par_vec(par_b, b, test_rhs);
            eigen_to_hypre_par_vec(par_x, x, test_result);

            HYPRE_BoomerAMGSetup(test_precond, A_par_, par_b, par_x);
            HYPRE_BoomerAMGSolve(test_precond, A_par_, par_b, par_x);
            hypre_vec_to_eigen(x, test_result);

            for (int i = 0; i < rhs.size(); ++i)
            {
                const double denom = std::abs(start_result(i));
                mags(i) = denom > 0 ? std::abs(test_result(i) / denom) : std::abs(test_result(i));
            }

            HYPRE_IJVectorDestroy(x);
            HYPRE_IJVectorDestroy(b);
            HYPRE_BoomerAMGDestroy(test_precond);
        }

        if (select_bad_dofs_from_row_norms_)
        {
            for (int i = 0; i < rhs.size(); ++i)
                mags(i) = A_row_.row(i).norm();
        }

        if (save_selection_criteria_)
        {
            std::ofstream file("criteria.txt", std::ios_base::app);
            file << mags.transpose() << "\n";
        }

        Eigen::VectorXd sorted = mags;
        std::sort(sorted.data(), sorted.data() + sorted.size());

        const double frac = std::min(std::max(bad_dof_threshold_, 0.0), 1.0);
        const int cutoff_index = static_cast<int>(std::floor(sorted.size() * (1.0 - frac)));
        const int clamped_cutoff = std::min(std::max(cutoff_index, 0), static_cast<int>(sorted.size() - 1));
        const double cutoff = sorted(clamped_cutoff);

        if (cutoff > 0)
        {
            for (int i = 0; i < rhs.size(); ++i)
            {
                if (mags(i) >= cutoff)
                    bad_indices_[0].insert(i);
            }
        }

        if (save_selected_indices_)
        {
            std::ofstream file("selected_indices.txt", std::ios_base::app);
            for (const auto idx : bad_indices_[0])
                file << idx << " ";
            file << "\n";
        }
    }

    void ExperimentalSolver::factorize_submatrices()
    {
        D_solvers_.clear();
        index_mappings_.clear();

        if (!do_mixed_precond_ || bad_indices_.empty())
            return;

        D_solvers_.resize(bad_indices_.size());
        index_mappings_.resize(bad_indices_.size());

        for (size_t i = 0; i < bad_indices_.size(); ++i)
        {
#ifdef POLYSOLVE_WITH_MKL
            D_solvers_[i] = std::make_unique<EigenWrapper<Eigen::PardisoLDLT<Eigen::SparseMatrix<double>>>>();
#else
            D_solvers_[i] = std::make_unique<EigenWrapper<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>>();
#endif
        }

        for (size_t i = 0; i < bad_indices_.size(); ++i)
        {
            int counter = 0;
            for (const int g : bad_indices_[i])
            {
                index_mappings_[i][g] = counter++;
            }
        }

        bad_indices_arrays_.clear();
        bad_indices_arrays_.resize(bad_indices_.size());
        for (size_t i = 0; i < bad_indices_.size(); ++i)
        {
            bad_indices_arrays_[i].assign(bad_indices_[i].begin(), bad_indices_[i].end());
        }

#pragma omp parallel for
        for (int si = 0; si < static_cast<int>(bad_indices_.size()); ++si)
        {
            const auto &subdomain = bad_indices_[static_cast<size_t>(si)];
            Eigen::SparseMatrix<double> D;
            D.resize(static_cast<int>(subdomain.size()), static_cast<int>(subdomain.size()));
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(subdomain.size() * 8);

            for (const int row_g : subdomain)
            {
                for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A_row_, row_g); it; ++it)
                {
                    const auto col_it = index_mappings_[static_cast<size_t>(si)].find(it.col());
                    if (col_it != index_mappings_[static_cast<size_t>(si)].end())
                    {
                        triplets.emplace_back(
                            index_mappings_[static_cast<size_t>(si)].at(it.row()),
                            col_it->second,
                            it.value());
                    }
                }
            }

            D.setFromTriplets(triplets.begin(), triplets.end());

            if (project_d_option_ != 0 && D.rows() > 0)
            {
                // Option 1: symmetrize D (cheap)
                if (project_d_option_ == 1)
                {
                    Eigen::SparseMatrix<double> Dt = D.transpose();
                    D = 0.5 * (D + Dt);
                }
                // Option 2: project onto PSD cone (dense) - expensive
                else if (project_d_option_ == 2)
                {
                    Eigen::MatrixXd dense = Eigen::MatrixXd(D);
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(dense);
                    if (eig.info() == Eigen::Success)
                    {
                        Eigen::VectorXd evals = eig.eigenvalues();
                        for (int i = 0; i < evals.size(); ++i)
                            evals(i) = std::max(0.0, evals(i));
                        dense = eig.eigenvectors() * evals.asDiagonal() * eig.eigenvectors().transpose();
                        D = dense.sparseView();
                    }
                }
            }

            D_solvers_[static_cast<size_t>(si)]->compute(D);
        }
    }

    void ExperimentalSolver::prepare_dss(const Eigen::VectorXd &rhs, const HYPRE_Solver &amg_precond)
    {
        select_bad_indices(rhs, amg_precond);

        if (decompose_subdomains_ && !bad_indices_.empty() && !bad_indices_[0].empty())
        {
            std::vector<int> all_bad;
            all_bad.reserve(bad_indices_[0].size());
            std::vector<int> global_to_local(A_row_.rows(), -1);
            for (const int g : bad_indices_[0])
            {
                global_to_local[g] = static_cast<int>(all_bad.size());
                all_bad.push_back(g);
            }

            DisjointSet ds(static_cast<int>(all_bad.size()));
            for (const int g : all_bad)
            {
                for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A_row_, g); it; ++it)
                {
                    const int lc = global_to_local[it.col()];
                    if (lc != -1)
                    {
                        ds.union_set(global_to_local[it.row()], lc);
                    }
                }
            }

            std::unordered_map<int, std::vector<int>> groups;
            groups.reserve(all_bad.size());
            for (const int g : all_bad)
            {
                groups[ds.find_set(global_to_local[g])].push_back(g);
            }

            bad_indices_.clear();
            bad_indices_.reserve(groups.size());
            for (auto &kv : groups)
            {
                bad_indices_.emplace_back(kv.second.begin(), kv.second.end());
            }
        }

        factorize_submatrices();

        if (print_conditioning_)
        {
            logger().warn("ExperimentalSolver: print_conditioning requested, but conditioning checks are not implemented in this port");
        }
    }

    // -----------------------------------------------------------------------------

    void ExperimentalSolver::pcg_solve(const Ref<const VectorXd> rhs, Ref<VectorXd> result, const HYPRE_Solver &amg_precond)
    {
        iteration_residual_norm_.clear();
        iteration_time_s_.clear();
        num_iterations_ = 0;

        const double bi_prod = rhs.dot(rhs);
        if (bi_prod <= 0.0)
        {
            result.setZero();
            num_iterations_ = 0;
            final_res_norm_ = 0;
            iteration_residual_norm_.push_back(0.0);
            iteration_time_s_.push_back(0.0);
            return;
        }
        const double normb = std::sqrt(bi_prod);

        Eigen::VectorXd A_times_result;
        matmul(result, A_times_result);
        Eigen::VectorXd r = rhs - A_times_result;
        iteration_residual_norm_.push_back(r.norm());
        iteration_time_s_.push_back(0.0);

        Eigen::VectorXd p(r.size());
        Eigen::VectorXd z(r.size());
        p.setZero();
        z.setZero();

        custom_mixed_precond_iter(amg_precond, r, z);
        p = z;

        double gamma = r.dot(z);
        double old_gamma = gamma;

        const double eps_sq = conv_tol_ * conv_tol_;

        for (int k = 0; k < max_iter_; ++k)
        {
            const auto iter_start = Clock::now();

            matmul(p, A_times_result);
            const double sdotp = p.dot(A_times_result);
            if (sdotp == 0.0)
            {
                logger().debug("Experimental PCG: zero sdotp, breaking");
                break;
            }

            const double alpha = gamma / sdotp;
            if (!(alpha > 0.0))
            {
                logger().debug("Experimental PCG: non-positive alpha, breaking (gamma={}, sdotp={})", gamma, sdotp);
                break;
            }

            result += alpha * p;
            r -= alpha * A_times_result;

            const double rnorm = r.norm();
            const double residual = use_absolute_tol_ ? rnorm : (rnorm / normb);

            const double rnorm_sq = residual * residual;
            if (rnorm_sq < eps_sq)
            {
                iteration_residual_norm_.push_back(rnorm);
                iteration_time_s_.push_back(seconds_since(iter_start));

                num_iterations_ = static_cast<HYPRE_Int>(k + 1);
                break;
            }

            z.setZero();
            custom_mixed_precond_iter(amg_precond, r, z);

            gamma = r.dot(z);
            const double beta = gamma / old_gamma;
            old_gamma = gamma;
            p = z + beta * p;

            num_iterations_ = static_cast<HYPRE_Int>(k + 1);

            iteration_residual_norm_.push_back(rnorm);
            iteration_time_s_.push_back(seconds_since(iter_start));
        }

        Eigen::VectorXd Ax;
        matmul(result, Ax);
        final_res_norm_ = (rhs - Ax).norm();
    }

    // -----------------------------------------------------------------------------

    void ExperimentalSolver::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        if (!has_matrix_)
        {
            throw std::runtime_error("ExperimentalSolver::factorize must be called before solve");
        }
        if (rhs.size() != result.size())
        {
            throw std::runtime_error("ExperimentalSolver::solve size mismatch");
        }

        if (save_problem_)
        {
            std::ofstream file("rhs.mat", std::ios_base::app);
            file << rhs.transpose() << "\n";
        }

        HYPRE_ParVector par_b;
        HYPRE_IJVector b;
        HYPRE_ParVector par_x;
        HYPRE_IJVector x;

        eigen_to_hypre_par_vec(par_b, b, rhs);
        eigen_to_hypre_par_vec(par_x, x, result);

        HYPRE_Solver amg_precond;
        HYPRE_BoomerAMGCreate(&amg_precond);
        HypreBoomerAMG_SetDefaultOptions(amg_precond);
        if (dimension_ > 1)
        {
            HypreBoomerAMG_SetElasticityOptions(amg_precond);
        }

        HYPRE_BoomerAMGSetup(amg_precond, A_par_, par_b, par_x);

        prepare_dss(rhs, amg_precond);
        pcg_solve(rhs, result, amg_precond);

        HYPRE_BoomerAMGDestroy(amg_precond);
        HYPRE_IJVectorDestroy(x);
        HYPRE_IJVectorDestroy(b);
    }

} // namespace polysolve::linear

#endif
