
////////////////////////////////////////////////////////////////////////////////
#include "ExperimentalSolver.hpp"

#include <HYPRE_krylov.h>
#include <HYPRE_utilities.h>

#ifdef POLYSOLVE_WITH_ICHOL
#include "cholesky.h"
#include "chol_hierarchy.h"
#endif
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cstdlib>

namespace polysolve::linear
{

    ////////////////////////////////////////////////////////////////////////////////

    ExperimentalSolver::ExperimentalSolver()
    {
        precond_num_ = 0;
        const char* num_threads_val = std::getenv("OMP_NUM_THREADS");
        if (num_threads_val)
        {
            num_threads = std::stoi(num_threads_val);
        }
        Eigen::setNbThreads(num_threads);
        logger->trace("Num Threads for ExperimentalSolver: {}", num_threads);
    }

    // Set solver parameters
    void ExperimentalSolver::set_parameters(const json &params)
    {
        if (params.contains("Hypre"))
        {
            if (params["Experimental"].contains("max_iter"))
            {
                max_iter_ = params["Experimental"]["max_iter"];
            }
            if (params["Experimental"].contains("pre_max_iter"))
            {
                pre_max_iter_ = params["Experimental"]["pre_max_iter"];
            }
            if (params["Experimental"].contains("tolerance"))
            {
                conv_tol_ = params["Experimental"]["tolerance"];
            }
            if (params["Experimental"].contains("theta"))
            {
                theta = params["Experimental"]["theta"];
            }
            if (params["Experimental"].contains("nodal_coarsening"))
            {
                nodal_coarsening = params["Experimental"]["nodal_coarsening"];
            }
            if (params["Experimental"].contains("interp_rbms"))
            {
                interp_rbms = params["Experimental"]["interp_rbms"];
            }
            if (params["Experimental"].contains("dimension"))
            {
                dimension_ = params["Experimental"]["dimension"];
            }
            if (params["Experimental"].contains("do_mixed_precond"))
            {
                do_mixed_precond = params["Experimental"]["do_mixed_precond"];
            }
            if (params["Experimental"].contains("print_conditioning"))
            {
                print_conditioning = params["Experimental"]["print_conditioning"];
            }
            if (params["Experimental"].contains("dss_in_middle"))
            {
                dss_in_middle = params["Experimental"]["dss_in_middle"];
            }
            if (params["Experimental"].contains("use_absolute_tol"))
            {
                use_absolute_tol = params["Experimental"]["use_absolute_tol"];
            }
            if (params["Experimental"].contains("select_bad_dofs_from_rhs"))
            {
                select_bad_dofs_from_rhs = params["Experimental"]["select_bad_dofs_from_rhs"];
            }
            if (params["Experimental"].contains("bad_dof_grad_threshold"))
            {
                bad_dof_grad_threshold = params["Experimental"]["bad_dof_grad_threshold"];
            }
#ifdef POLYSOLVE_WITH_ICHOL
            if (params["Experimental"].contains("use_incomplete_cholesky_precond"))
            {
                use_incomplete_cholesky_precond = params["Experimental"]["use_incomplete_cholesky_precond"];
            }
            if (params["Experimental"].contains("rho"))
            {
                rho = params["Experimental"]["rho"];
            }
#endif
        }
    }

    void ExperimentalSolver::get_info(json &params) const
    {
        params["num_iterations"] = num_iterations;
        params["final_res_norm"] = final_res_norm;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void ExperimentalSolver::factorize(const StiffnessMatrix &Ain)
    {
        assert(precond_num_ > 0);

        sparse_A = Ain;

        #ifdef POLYSOLVE_WITH_ICHOL
        if (use_incomplete_cholesky_precond)
        {
            logger->trace("Factorizing for ichol");
            pt.put<double>("nei_num.value", rho);
            pt.put<double>("alpha.value", 1e-4);
            pt.put<std::ptrdiff_t>("max_su_size.value", 64);
            pt.put<int>("num_threads.value", num_threads);
            pt.put<int>("subst_num_threads.value", num_threads);
            
            Eigen::Matrix<size_t, -1, -1> test_elements = elements_.cast<size_t>();
            mschol::chol_hierarchy builder(test_elements, positions_, "tets");
            
            std::vector<std::shared_ptr<mschol::chol_level>> levels;
            builder.build(levels, 125, dimension_);
            builder.get_dof_remapping(ichol_dof_remapping);

            sparse_A.resize(Ain.rows(), Ain.cols());
            sparse_A.data().squeeze();

            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(Ain.nonZeros());

            for (int k = 0; k < Ain.outerSize(); ++k)
            {
                for (StiffnessMatrix::InnerIterator it(Ain, k); it; ++it)
                {
                    triplets.push_back(Eigen::Triplet<double>(remap_dof(it.row()), remap_dof(it.col()), it.value()));
                }
            }

            sparse_A.setFromTriplets(triplets.begin(), triplets.end());

            std::ofstream file("A.mat");
            file << sparse_A;
            file.close();

            inc_chol_precond = std::make_shared<mschol::ichol_precond>(levels, pt);
            inc_chol_precond->analyse_pattern(sparse_A);
            inc_chol_precond->factorize(sparse_A);
        }
#endif

        if (print_conditioning)
        {
            Eigen::BDCSVD<Eigen::MatrixXd> svd(sparse_A);
            double cond = svd.singularValues()(0) 
            / svd.singularValues()(svd.singularValues().size()-1);

            logger->debug("Condition number: {}", cond);
        }

        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }

        has_matrix_ = true;
        const HYPRE_Int rows = sparse_A.rows();
        const HYPRE_Int cols = sparse_A.cols();

        HYPRE_IJMatrixCreate(0, 0, rows - 1, 0, cols - 1, &A);
        // HYPRE_IJMatrixSetPrintLevel(A, 2);
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A);

        // TODO: More efficient initialization of the Hypre matrix?
        for (HYPRE_Int k = 0; k < Ain.outerSize(); ++k)
        {
            for (StiffnessMatrix::InnerIterator it(Ain, k); it; ++it)
            {
                const HYPRE_Int i[1] = {it.row()};
                const HYPRE_Int j[1] = {it.col()};
                const HYPRE_Complex v[1] = {it.value()};
                HYPRE_Int n_cols[1] = {1};

                HYPRE_IJMatrixSetValues(A, 1, n_cols, i, j, v);
            }
        }

        HYPRE_IJMatrixAssemble(A);
        HYPRE_IJMatrixGetObject(A, (void **)&parcsr_A);

        assert(bad_indices_.size() == 1);
        auto &subdomain = bad_indices_[0];

        // Save submatrix for direct step. TODO: refactor for multiple subdomains
        if (!do_mixed_precond || subdomain.size() == 0 || select_bad_dofs_from_rhs)
        {
            return;
        }

        factorize_submatrix(subdomain);
    }

    ////////////////////////////////////////////////////////////////////////////////

    namespace
    {

        void eigen_to_hypre_par_vec(HYPRE_ParVector &par_x, HYPRE_IJVector &ij_x, const Eigen::VectorXd &x)
        {
    #ifdef HYPRE_WITH_MPI
            HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, x.size() - 1, &ij_x);
    #else
            HYPRE_IJVectorCreate(0, 0, x.size() - 1, &ij_x);
    #endif
            HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
            HYPRE_IJVectorInitialize(ij_x);

            for (HYPRE_Int i = 0; i < x.size(); ++i)
            {
                const HYPRE_Int index[1] = {i};
                const HYPRE_Complex v[1] = {HYPRE_Complex(x(i))};
                HYPRE_IJVectorSetValues(ij_x, 1, index, v);
            }

            HYPRE_IJVectorAssemble(ij_x);
            HYPRE_IJVectorGetObject(ij_x, (void **)&par_x);
        }

        void hypre_vec_to_eigen(const HYPRE_IJVector &ij_x, Eigen::VectorXd &x)
        {
            for (HYPRE_Int i = 0; i < x.size(); ++i)
            {
                const HYPRE_Int index[1] = {i};
                HYPRE_Complex v[1];
                HYPRE_IJVectorGetValues(ij_x, 1, index, v);

                x(i) = v[0];
            }
        }

        void calculate_rbms(Eigen::VectorXd &rbm_xy, Eigen::VectorXd &rbm_zx, Eigen::VectorXd &rbm_yz, const Eigen::MatrixXd &positions, const int dim)
        {
            rbm_xy.resize(positions.size());
            rbm_xy.setZero();
            
            if (dim == 3)
            {
                rbm_zx.resize(positions.size());
                rbm_yz.resize(positions.size());
            
                rbm_zx.setZero();
                rbm_yz.setZero();
            }

            for (int i = 0; i < positions.rows(); ++i)
            {
                rbm_xy(0 + i*dim) = positions(i, 1);
                rbm_xy(1 + i*dim) = -1 * positions(i, 0);

                if (dim == 3)
                {
                    rbm_zx(1 + i*dim) = positions(i, 2);
                    rbm_zx(2 + i*dim) = -1 * positions(i, 1);

                    rbm_yz(2 + i*dim) = positions(i, 0);
                    rbm_yz(0 + i*dim) = -1 * positions(i, 2);
                }
            }
        }

        void HypreBoomerAMG_SetDefaultOptions(HYPRE_Solver &amg_precond)
        {
            // AMG coarsening options:
            int coarsen_type = 10; // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
            int agg_levels = 1;    // number of aggressive coarsening levels
            double theta = 0.25;   // strength threshold: 0.25, 0.5, 0.8

            // AMG interpolation options:
            int interp_type = 6; // 6 = extended+i, 0 = classical
            int Pmax = 4;        // max number of elements per row in P

            // AMG relaxation options:
            int relax_type = 8;   // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
            int relax_sweeps = 1; // relaxation sweeps on each level

            // Additional options:
            int print_level = 0; // print AMG iterations? 1 = no, 2 = yes
            int max_levels = 25; // max number of levels in AMG hierarchy

            HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
            HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);
            HYPRE_BoomerAMGSetNumSweeps(amg_precond, relax_sweeps);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
            HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type);
            HYPRE_BoomerAMGSetPMaxElmts(amg_precond, Pmax);
            HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level);
            HYPRE_BoomerAMGSetMaxLevels(amg_precond, max_levels);

            // Use as a preconditioner (one V-cycle, zero tolerance)
            HYPRE_BoomerAMGSetMaxIter(amg_precond, 1);
            HYPRE_BoomerAMGSetTol(amg_precond, 0.0);
        }

        void HypreBoomerAMG_SetElasticityOptions(HYPRE_Solver &amg_precond, int dim, double theta, bool nodal_coarsening, bool interp_rbms, const Eigen::MatrixXd &positions, std::vector<int> &dof_to_function, std::vector<HYPRE_IJVector> &rbms, std::vector<HYPRE_ParVector> &par_rbms)
        {
            // Make sure the systems AMG options are set
            HYPRE_BoomerAMGSetNumFunctions(amg_precond, dim);

            //HYPRE_BoomerAMGSetDofFunc(amg_precond, (HYPRE_Int*) dof_to_function.data());

            // More robust options with respect to convergence
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);

            // Nodal coarsening options (nodal coarsening is required for this solver)
            // See hypre's new_ij driver and the paper for descriptions.
            int nodal = 4;        // strength reduction norm: 1, 3 or 4
            int nodal_diag = 1;   // diagonal in strength matrix: 0, 1 or 2
            int relax_coarse = 8; // smoother on the coarsest grid: 8, 99 or 29

            // Elasticity interpolation options
            int interp_vec_variant = 2;    // 1 = GM-1, 2 = GM-2, 3 = LN
            int q_max = 4;                 // max elements per row for each Q
            int smooth_interp_vectors = 1; // smooth the rigid-body modes?

            // Optionally pre-process the interpolation matrix through iterative weight
            // refinement (this is generally applicable for any system)
            int interp_refine = 1;

            if (nodal_coarsening) 
            {
                HYPRE_BoomerAMGSetNodal(amg_precond, nodal);
                HYPRE_BoomerAMGSetNodalDiag(amg_precond, nodal_diag);
                HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_coarse, 3);
            }

            if (interp_rbms)
            {
                if (dim != 2 && dim != 3)
                {
                    assert(false);
                }

                HYPRE_BoomerAMGSetInterpVecVariant(amg_precond, interp_vec_variant);
                HYPRE_BoomerAMGSetInterpVecQMax(amg_precond, q_max);

                // HYPRE_BoomerAMGSetSmoothInterpVectors(amg_precond, smooth_interp_vectors);
                // HYPRE_BoomerAMGSetInterpRefine(amg_precond, interp_refine);

                Eigen::VectorXd rbm_xy, rbm_zx, rbm_yz;
                calculate_rbms(rbm_xy, rbm_zx, rbm_yz, positions, dim);

                eigen_to_hypre_par_vec(par_rbms[0], rbms[0], rbm_xy);
                if (dim == 3)
                {
                    eigen_to_hypre_par_vec(par_rbms[1], rbms[1], rbm_zx);
                    eigen_to_hypre_par_vec(par_rbms[2], rbms[2], rbm_yz);
                }
            
                HYPRE_BoomerAMGSetInterpVectors(amg_precond, par_rbms.size(), &(par_rbms[0]));
            }
        }

    } // anonymous namespace


    ////////////////////////////////////////////////////////////////////////////////

    void ExperimentalSolver::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        
        assert(result.size() == rhs.size());
        Eigen::VectorXd remapped_rhs = rhs;
        Eigen::VectorXd remapped_result = result;

#ifdef POLYSOLVE_WITH_ICHOL
        if (use_incomplete_cholesky_precond)
        {
            for (int i = 0; i < rhs.size(); ++i)
            {
                remapped_rhs(i) = rhs(remap_dof(i));
                remapped_result(i) = result(remap_dof(i));
            }
        }
#endif

        HYPRE_ParVector par_b;
        HYPRE_ParVector par_x;
        HYPRE_IJVector x;
        HYPRE_IJVector b;

        {
            POLYSOLVE_SCOPED_STOPWATCH("copy x and b", copy_b_and_x_time, *logger);
            eigen_to_hypre_par_vec(par_b, b, remapped_rhs);
            eigen_to_hypre_par_vec(par_x, x, remapped_result);
        }

        /* AMG preconditioner */
        HYPRE_Solver precond;

        /* Now set up the AMG preconditioner and specify any parameters */
        HYPRE_BoomerAMGCreate(&precond);

        const int num_rbms = dimension_ == 2 ? 1 : 3;
        std::vector<HYPRE_ParVector> par_rbms(num_rbms);
        std::vector<HYPRE_IJVector> rbms(num_rbms);
        
        {
            POLYSOLVE_SCOPED_STOPWATCH("set options", set_options_time, *logger);
            HypreBoomerAMG_SetDefaultOptions(precond);
            if (dimension_ > 1)
            {
                HypreBoomerAMG_SetElasticityOptions(
                    precond, 
                    dimension_, 
                    theta, 
                    nodal_coarsening, 
                    interp_rbms, 
                    positions_,
                    dof_to_function_, 
                    rbms, 
                    par_rbms
                );
            }
        }

        if (select_bad_dofs_from_rhs)
        {
            bad_indices_.clear();
            bad_indices_.resize(1);
            auto &subdomain = bad_indices_[0];
            {
                POLYSOLVE_SCOPED_STOPWATCH("bad dof selection time", bad_dof_selection_time, *logger);

                assert(rhs.size() % dimension_ == 0);
                Eigen::VectorXd sq_mags(rhs.size() / dimension_);
                for (int i = 0; i < rhs.size() / dimension_; ++i)
                {
                    double sq_mag = rhs(dimension_ * i) * rhs(dimension_ * i);
                    for (int j = 1; j < dimension_; ++j)
                    {
                        sq_mag += rhs(dimension_ * i + j) * rhs(dimension_ * i + j);
                    } 
                    sq_mags(i) = sq_mag;
                }
                Eigen::VectorXd sq_mags_copy = sq_mags;
                std::sort(sq_mags_copy.data(), sq_mags_copy.data() + sq_mags_copy.size());
                const int cutoff_index = sq_mags_copy.size() * (1 - bad_dof_grad_threshold);
                const double cutoff = sq_mags_copy(cutoff_index);

                if (cutoff > 0)
                {
                    for (int i = 0; i < rhs.size() / dimension_; ++i)
                    {
                        if (sq_mags(i) >= cutoff)
                        {
                            for (int j = 0; j < dimension_; ++j)
                            {
                                subdomain.insert(dimension_ * i + j);
                            }
                        }
                    }
                }
            }

            factorize_submatrix(subdomain);
        }

        /* Now setup and solve! */
        {
            POLYSOLVE_SCOPED_STOPWATCH("actual solve time", actual_solve_time, *logger);

            /* Custom PCG */

            double bi_prod = remapped_rhs.dot(remapped_rhs);
            logger->trace("Experimental solver bi prod: {}", bi_prod);

            double eps;

            if (bi_prod > 0.0)
            {
                eps = conv_tol_ * conv_tol_;
            }
            else 
            {
                result.setZero();
                num_iterations = 0;
                final_res_norm = 0;
                logger->debug("Experimental solver Iterations: {}", num_iterations);
                logger->debug("Experimental solver Final Relative Residual Norm: {}", final_res_norm);
                return;
            }

            Eigen::VectorXd r = remapped_rhs - (sparse_A * remapped_result);

            Eigen::VectorXd p(r.size());
            Eigen::VectorXd z(r.size());
            p.setZero();
            z.setZero();

            HYPRE_BoomerAMGSetup(precond, parcsr_A, par_b, par_x);

#ifdef POLYSOLVE_WITH_ICHOL
            if (use_incomplete_cholesky_precond)
            {
                z = inc_chol_precond->solve(r);
            } else
#endif
            if (!do_mixed_precond || bad_indices_.size() == 0)
            {
                amg_precond_iter(precond, r, z);
            }
            else
            {
                custom_mixed_precond_iter(precond, r, z);
            }
            
            p = z;

            double gamma = r.dot(z);
            double old_gamma = gamma;

            for (int k = 0; k < max_iter_; ++k)
            {
                num_iterations = k + 1;

                double sdotp = p.dot(sparse_A * p);

                if (sdotp == 0.0)
                {
                    logger->debug("Experimental solver error: zero sdotp value");
                    break;
                }

                double alpha = gamma / sdotp;

                if (alpha <= 0.0)
                {
                    logger->debug("Experimental solver error: negative or zero alpha value");
                    break;
                } 
                else if (alpha < __DBL_MIN__)
                {
                    logger->debug("Experimental solver error: subnormal alpha value");
                    break;
                }

                remapped_result += alpha * p;
                r -= alpha * sparse_A * p;
                //r = rhs - (sparse_A * result);
                double drob2 = alpha * alpha * p.dot(p);
                if (!use_absolute_tol) 
                {
                    drob2 /= bi_prod;
                }

                if (drob2 < conv_tol_ * conv_tol_)
                {
                    logger->debug("Experimental solver converged: change in residual too small");
                    break;
                }

                double i_prod = r.dot(r);
                logger->trace("Experimental solver i prod: {}", i_prod);
                if (!use_absolute_tol) 
                {
                    i_prod /= bi_prod;
                }

                if (i_prod < eps)
                {
                    logger->debug("Experimental solver converged: residual too small");
                    break;
                }

                z.setZero(); 

#ifdef POLYSOLVE_WITH_ICHOL
                if (use_incomplete_cholesky_precond)
                {
                    z = inc_chol_precond->solve(r);
                } else
#endif
                if (!do_mixed_precond || bad_indices_.size() == 0)
                {
                    amg_precond_iter(precond, r, z);
                }
                else
                {
                    custom_mixed_precond_iter(precond, r, z);
                }

                gamma = r.dot(z);
                double beta = gamma / old_gamma;
                old_gamma = gamma;

                p = z + beta*p;
            }

            final_res_norm = (remapped_rhs - (sparse_A * remapped_result)).norm();
        }

        logger->debug("Experimental solver Iterations: {}", num_iterations);
        logger->debug("Experimental solver Final Relative Residual Norm: {}", final_res_norm);

        result = remapped_result;

#ifdef POLYSOLVE_WITH_ICHOL
        if (use_incomplete_cholesky_precond)
        {
            for (int i = 0; i < result.size(); ++i)
            {
                result(remap_dof(i)) = remapped_result(i);
            }
        }
#endif

        /* Destroy preconditioner */
        HYPRE_BoomerAMGDestroy(precond);

        HYPRE_IJVectorDestroy(x);
        HYPRE_IJVectorDestroy(b);

        if (select_bad_dofs_from_rhs)
        {
            bad_indices_.clear();
        }

    }

    void ExperimentalSolver::custom_mixed_precond_iter(const HYPRE_Solver &precond, const Eigen::VectorXd &r, Eigen::VectorXd &z)
    {        
        Eigen::VectorXd z1(r.size());
        Eigen::VectorXd z2(r.size());
        Eigen::VectorXd z3(r.size());
        z1.setZero();
        z2.setZero();
        z3.setZero();

        //amg_precond_iter(precond, r, z1);

        assert(bad_indices_.size() == 1);
        if (bad_indices_[0].size() == 0)
        {
            amg_precond_iter(precond, r, z1);
            z = z1;
            return;
        }

        if (dss_in_middle)
        {
            amg_precond_iter(precond, r, z1);
            dss_precond_iter(z1, r, z2);
            amg_precond_iter(precond, r - sparse_A * z2, z3);
            z = z2 + z3;
        }
        else
        {
            Eigen::VectorXd z0(r.size());
            z0.setZero();
            dss_precond_iter(z0, r, z1);
            amg_precond_iter(precond, r - sparse_A * z1, z2);
            z2 += z1;
            dss_precond_iter(z2, r, z);
        }

    }

    void ExperimentalSolver::amg_precond_iter(const HYPRE_Solver &precond, const Eigen::Ref<const VectorXd> eigen_b, Eigen::VectorXd &eigen_x)
    {
        HYPRE_ParVector par_x;
        HYPRE_IJVector x;
        HYPRE_ParVector par_b;
        HYPRE_IJVector b;

        eigen_to_hypre_par_vec(par_x, x, eigen_x);
        eigen_to_hypre_par_vec(par_b, b, eigen_b);

        HYPRE_BoomerAMGSolve(precond, parcsr_A, par_b, par_x);

        hypre_vec_to_eigen(x, eigen_x);
    }

    void ExperimentalSolver::dss_precond_iter(const Eigen::VectorXd &z, const Eigen::VectorXd &r, Eigen::VectorXd &next_z)
    {
        logger->trace("DSS Step");

        auto &subdomain = bad_indices_[0];

        Eigen::VectorXd sub_rhs;
        Eigen::VectorXd sub_result;
        sub_rhs.resize(subdomain.size());
        sub_result.resize(subdomain.size());

        int i_counter = 0;
        for (auto &i : subdomain)
        {
            sub_rhs(i_counter) = r(i) - sparse_A.row(i).dot(z);
            ++i_counter;
        }

        sub_result = D_solver.solve(sub_rhs);
        i_counter = 0;
        next_z = z;
        for (auto &i : subdomain)
        {
            next_z(i) += sub_result(i_counter);
            ++i_counter;
        }
    }

#ifdef POLYSOLVE_WITH_ICHOL
    int ExperimentalSolver::remap_dof(const int index)
    {
        int nod_index = index / dimension_;
        int func_offset = index % dimension_;
        return dimension_ * ichol_dof_remapping(nod_index) + func_offset;
    }
#endif

    void ExperimentalSolver::factorize_submatrix(const std::set<int> subdomain)
    {
        Eigen::MatrixXd D;
        D.resize(subdomain.size(), subdomain.size());

        logger->debug("Subdomain size: {}", subdomain.size());

        {
            POLYSOLVE_SCOPED_STOPWATCH("assemble D", dss_assembly_time, *logger);
            int i_counter = 0;
            for (auto i : subdomain)
            {
                int j_counter = 0;
                for (auto j : subdomain)
                {
                    D(i_counter, j_counter) = sparse_A.coeff(i, j);
                    ++j_counter;
                }
                ++i_counter;
            }
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("factorize D", dss_factorization_time, *logger);
            D_solver.compute(D.sparseView());
        }
    }

    ////////////////////////////////////////////////////////////////////////////////

    ExperimentalSolver::~ExperimentalSolver()
    {
        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }
    }

} // namespace polysolve::linear

