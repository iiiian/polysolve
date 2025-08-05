
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
#include <unordered_map>

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

#ifdef HYPRE_WITH_MPI
        int done_already;

        MPI_Initialized(&done_already);
        if (!done_already)
        {
            /* Initialize MPI */
            int argc = 1;
            char name[] = "";
            char *argv[] = {name};
            char **argvv = &argv[0];
            MPI_Init(&argc, &argvv);
            MPI_Comm_rank(MPI_COMM_WORLD, &myid);
            MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        if (myid == 0)
        {
            Eigen::setNbThreads(num_threads);
        }
        else 
        {
            Eigen::setNbThreads(1);
        }
#else
        Eigen::setNbThreads(num_threads);
#endif
    }

    // Set solver parameters
    void ExperimentalSolver::set_parameters(const json &params)
    {
        if (params.contains("Experimental"))
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
            if (params["Experimental"].contains("select_bad_dofs_from_row_norms"))
            {
                select_bad_dofs_from_row_norms = params["Experimental"]["select_bad_dofs_from_row_norms"];
            }
            if (params["Experimental"].contains("select_bad_dofs_from_amg"))
            {
                select_bad_dofs_from_amg = params["Experimental"]["select_bad_dofs_from_amg"];
            }
            if (params["Experimental"].contains("bad_dof_threshold"))
            {
                bad_dof_threshold = params["Experimental"]["bad_dof_threshold"];
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
            if (params["Experimental"].contains("save_selection_criteria"))
            {
                save_selection_criteria = params["Experimental"]["save_selection_criteria"];
            }
            if (params["Experimental"].contains("save_problem"))
            {
                save_problem = params["Experimental"]["save_problem"];
            }
            if (params["Experimental"].contains("save_selected_indices"))
            {
                save_selected_indices = params["Experimental"]["save_selected_indices"];
            }
            if (params["Experimental"].contains("use_gmres"))
            {
                use_gmres = params["Experimental"]["use_gmres"];
            }
            if (params["Experimental"].contains("use_minres"))
            {
                use_minres = params["Experimental"]["use_minres"];
            }
            if (params["Experimental"].contains("m"))
            {
                m_ = params["Experimental"]["m"];
            }
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
#ifdef HYPRE_WITH_MPI
        if (myid == 0)
        {
            int start_factorize =1 ;
            MPI_Bcast(&start_factorize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        } 
        else 
        {
            double eigen_copy_time;
            {
                POLYSOLVE_SCOPED_STOPWATCH("eigen matrix copy time", eigen_copy_time, *logger);
                sparse_A = Ain;
            }
        }
#endif
        logger->trace("Num Threads for ExperimentalSolver: {}", num_threads);
        logger->trace("Eigen num threads: {}", Eigen::nbThreads());

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

            Eigen::VectorXi old_to_new(ichol_dof_remapping.size());
            for (int i = 0; i < ichol_dof_remapping.size(); ++i)
            {
                old_to_new(ichol_dof_remapping[i]) = i; 
            }

            for (int k = 0; k < Ain.outerSize(); ++k)
            {
                for (StiffnessMatrix::InnerIterator it(Ain, k); it; ++it)
                {   
                    int nod_index_i = it.row() / dimension_;
                    int func_offset_i = it.row() % dimension_;
                    int new_i = dimension_ * old_to_new(nod_index_i) + func_offset_i;

                    int nod_index_j = it.col() / dimension_;
                    int func_offset_j = it.col() % dimension_;
                    int new_j = dimension_ * old_to_new(nod_index_j) + func_offset_j;
                    
                    triplets.push_back(Eigen::Triplet<double>(new_i, new_j, it.value()));
                }
            }

            sparse_A.setFromTriplets(triplets.begin(), triplets.end());

            inc_chol_precond = std::make_shared<mschol::ichol_precond>(levels, pt);
            inc_chol_precond->analyse_pattern(sparse_A);
            inc_chol_precond->factorize(sparse_A);
        }
#endif

        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }

        if (save_problem && myid == 0)
        {
            logger->trace("Saving problem");
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(sparse_A.nonZeros());

            for (int k = 0; k < Ain.outerSize(); ++k)
            {
                for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sparse_A, k); it; ++it)
                {   
                    triplets.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
                }
            }

            std::ofstream file("A.mat", std::ios_base::app);
            file << sparse_A.rows() << " " << sparse_A.cols() << " " << sparse_A.nonZeros() << std::endl;
            for (auto &trip : triplets)
            {
                file << trip.row() << " " << trip.col() << " " << trip.value() << " ";
            }
            file << std::endl;
            file.close();
        }

        has_matrix_ = true;

        const HYPRE_Int rows = sparse_A.rows();
        const HYPRE_Int cols = sparse_A.cols();

        int local_size = rows / num_procs;
        start_i = myid == 0 ? 0 : local_size * myid + myid;
        end_i = myid == (num_procs - 1) ? rows - 1 : start_i + local_size;
        logger->trace("World size: {}, myid: {}", num_procs, myid);
        logger->trace("start {}, end {}", start_i, end_i);
#ifdef HYPRE_WITH_MPI
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, start_i, end_i, start_i, end_i, &A);
#else
        HYPRE_IJMatrixCreate(0, 0, rows - 1, 0, cols - 1, &A);
#endif
        // HYPRE_IJMatrixSetPrintLevel(A, 2);
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A);

        // TODO: More efficient initialization of the Hypre matrix?
        double matrix_copy_time;
        {
            POLYSOLVE_SCOPED_STOPWATCH("copy matrix time", matrix_copy_time, *logger);
            for (HYPRE_Int k = 0; k < sparse_A.outerSize(); ++k)
            {
                HYPRE_Int row[1]; 
                int counter = 0;
                std::vector<HYPRE_Int> cols;
                std::vector<double> vals;
                for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sparse_A, k); it; ++it)
                {
                    ++counter;
                    row[0] = it.row();
                    cols.push_back((HYPRE_Int)it.col());
                    vals.push_back(it.value());
                }
                HYPRE_Int n_cols[1] = {counter};
                HYPRE_IJMatrixSetValues(A, 1, n_cols, row, cols.data(), vals.data());
            }
            HYPRE_IJMatrixAssemble(A);
            HYPRE_IJMatrixGetObject(A, (void **)&parcsr_A);
        }

    }

    ////////////////////////////////////////////////////////////////////////////////

    namespace
    {

        void eigen_to_hypre_par_vec(HYPRE_ParVector &par_x, HYPRE_IJVector &ij_x, const Eigen::VectorXd &x, int start_i, int end_i)
        {
    #ifdef HYPRE_WITH_MPI
            HYPRE_IJVectorCreate(MPI_COMM_WORLD, start_i, end_i, &ij_x);
    #else
            HYPRE_IJVectorCreate(0, 0, x.size() - 1, &ij_x);
    #endif
            HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
            HYPRE_IJVectorInitialize(ij_x);

    #ifdef HYPRE_WITH_MPI
            HYPRE_IJVectorSetValues(ij_x, end_i - start_i + 1, nullptr, x.data() + start_i);
    #else
            HYPRE_IJVectorSetValues(ij_x, x.size(), nullptr, x.data());
    #endif

            HYPRE_IJVectorAssemble(ij_x);
            HYPRE_IJVectorGetObject(ij_x, (void **)&par_x);
        }

        void hypre_vec_to_eigen(const HYPRE_IJVector &ij_x, Eigen::Ref<Eigen::VectorXd> x, int start_i, int end_i)
        {
    #ifdef HYPRE_WITH_MPI
            x.setZero();
            HYPRE_IJVectorGetValues(ij_x, end_i - start_i + 1, nullptr, x.data() + start_i);
            MPI_Allreduce(MPI_IN_PLACE, x.data(), x.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #else
            HYPRE_IJVectorGetValues(ij_x, x.size(), nullptr, x.data());
    #endif            
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
            HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_type, 3);
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

                eigen_to_hypre_par_vec(par_rbms[0], rbms[0], rbm_xy, 0, positions.rows() - 1);
                if (dim == 3)
                {
                    eigen_to_hypre_par_vec(par_rbms[1], rbms[1], rbm_zx, 0, positions.rows() - 1);
                    eigen_to_hypre_par_vec(par_rbms[2], rbms[2], rbm_yz, 0, positions.rows() - 1);
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

#ifdef HYPRE_WITH_MPI
        if (myid == 0)
        {
            MPI_Bcast(remapped_rhs.data(), remapped_rhs.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(remapped_result.data(), remapped_result.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            int start_solve = 1;
            MPI_Bcast(&start_solve, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
#endif


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

        if (save_problem)
        {
            std::ofstream file("rhs.mat", std::ios_base::app);
            file << remapped_rhs.transpose();
            file << std::endl;
            file.close();
        }

        HYPRE_ParVector par_b;
        HYPRE_ParVector par_x;
        HYPRE_IJVector x;
        HYPRE_IJVector b;

        {
            POLYSOLVE_SCOPED_STOPWATCH("copy x and b", copy_b_and_x_time, *logger);
            eigen_to_hypre_par_vec(par_b, b, remapped_rhs, start_i, end_i);
            eigen_to_hypre_par_vec(par_x, x, remapped_result, start_i, end_i);
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

        select_bad_indices(remapped_rhs);
        factorize_submatrix();

        if (print_conditioning)
        {
            check_matrix_conditioning("Hessian", sparse_A);
            check_matrix_conditioning("Preconditioned Hessian", bad_indices_[0]);
        }

        #ifdef HYPRE_WITH_MPI
            MPI_Barrier(MPI_COMM_WORLD);
        #endif
            HYPRE_BoomerAMGSetup(precond, parcsr_A, par_b, par_x);

        Eigen::VectorXd test_u(remapped_rhs.size());
        Eigen::VectorXd test_v(remapped_rhs.size());
        test_u.setRandom();
        test_v.setRandom();
        test_u /= test_u.norm();
        test_v /= test_v.norm();
        Eigen::VectorXd u_amg = test_u;
        Eigen::VectorXd v_amg = test_v;
        Eigen::VectorXd u_custom = test_u;
        Eigen::VectorXd v_custom = test_v;
        amg_precond_iter(precond, test_u, u_amg);
        amg_precond_iter(precond, test_v, v_amg);
        custom_mixed_precond_iter(precond, test_u, u_custom);
        custom_mixed_precond_iter(precond, test_v, v_custom);
        double sym_check = test_v.dot(sparse_A * test_u) - test_u.dot(sparse_A * test_v);
        double sym_check2 = test_v.dot(u_amg) - test_u.dot(v_amg);
        double sym_check3 = test_v.dot(v_amg);
        double sym_check4 = test_u.dot(u_amg);
        double sym_check5 = test_v.dot(u_custom) - test_u.dot(v_custom);
        double sym_check6 = test_v.dot(v_custom);
        double sym_check7 = test_u.dot(u_custom);
        logger->trace("H Symmetry check: {}", sym_check);
        logger->trace("AMG Symmetry check: {}", sym_check2);
        logger->trace("AMG SPD check1: {}", sym_check3);
        logger->trace("AMG SPD check2: {}", sym_check4);
        logger->trace("Custom Symmetry check: {}", sym_check5);
        logger->trace("Custom SPD check1: {}", sym_check6);
        logger->trace("Custom SPD check2: {}", sym_check7);

        /* Now setup and solve! */
        {
#ifdef HYPRE_WITH_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            POLYSOLVE_SCOPED_STOPWATCH("actual solve time", actual_solve_time, *logger);

            if (use_minres)
            {
                minres_solve(remapped_rhs, remapped_result, par_b, par_x, precond);
            } 
            else if (use_gmres)
            {
                gmres_solve(remapped_rhs, remapped_result, par_b, par_x, precond);
            } 
            else
            {
                pcg_solve(remapped_rhs, remapped_result, par_b, par_x, precond);
            }

            Eigen::VectorXd A_times_result;
            matmul(remapped_result, sparse_A, A_times_result);
            final_res_norm = (remapped_rhs - A_times_result).norm();
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
        double destroy_time;
        {
            POLYSOLVE_SCOPED_STOPWATCH("destroy time", destroy_time, *logger);
            HYPRE_BoomerAMGDestroy(precond);

            HYPRE_IJVectorDestroy(x);
            HYPRE_IJVectorDestroy(b);
        }
    }

    void ExperimentalSolver::pcg_solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond)
    {
        double pre_loop_time;
        double bi_prod, eps, gamma, old_gamma;
        Eigen::VectorXd r, p, z;
        {
            POLYSOLVE_SCOPED_STOPWATCH("pre loop time: ", pre_loop_time, *logger);
        
            bi_prod = rhs.dot(rhs);
            logger->trace("Experimental solver bi prod: {}", bi_prod);

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

            Eigen::VectorXd A_times_result;
            matmul(result, sparse_A, A_times_result);
            r = rhs - A_times_result;

            p.resize(r.size());
            z.resize(r.size());
            p.setZero();
            z.setZero();

#ifdef POLYSOLVE_WITH_ICHOL
            if (use_incomplete_cholesky_precond)
            {
                z = inc_chol_precond->solve(r);
            } else
#endif
            {
                custom_mixed_precond_iter(precond, r, z);
            }
            
            
            p = z;

            gamma = r.dot(z);
            old_gamma = gamma;
        }

        double loop_time;
        for (int k = 0; k < max_iter_; ++k)
        {
            
            POLYSOLVE_SCOPED_STOPWATCH("main loop time: ", loop_time, *logger);
            num_iterations = k + 1;

            Eigen::VectorXd A_times_p;
            matmul(p, sparse_A, A_times_p);
            double sdotp = p.dot(A_times_p);

            if (sdotp == 0.0)
            {
                logger->debug("Experimental solver error: zero sdotp value");
                break;
            }

            double alpha = gamma / sdotp;

            if (alpha <= 0.0)
            {
                logger->debug("Experimental solver error: negative or zero alpha value. gamma: {}, sdotp: {}", gamma, sdotp);
                break;
            } 
            else if (alpha < __DBL_MIN__)
            {
                logger->debug("Experimental solver error: subnormal alpha value");
                break;
            }

            result += alpha * p;
            matmul(p, sparse_A, A_times_p);
            r -= alpha * A_times_p;
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
            {
                custom_mixed_precond_iter(precond, r, z);
            }

            gamma = r.dot(z);
            double beta = gamma / old_gamma;
            old_gamma = gamma;

            p = z + beta*p;
        }
    }

    void ExperimentalSolver::minres_solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond)
    {
        // TODO
        double beta, eta, gamma0, gamma1, sigma0, sigma1;
        double alpha, delta, rho1, rho2, rho3, norm_goal;

        Eigen::VectorXd v1;
        matmul(result, sparse_A, v1);
        v1 = rhs - v1;

        Eigen::VectorXd q(rhs.size());
        Eigen::VectorXd v0(rhs.size());
        Eigen::VectorXd w0(rhs.size());
        Eigen::VectorXd w1(rhs.size());
        Eigen::VectorXd u1(rhs.size());
        u1.setZero();
        custom_mixed_precond_iter(precond, v1, u1);

        /*
        if (!do_mixed_precond || bad_indices_.size() == 0 || bad_indices_[0].size() == 0)
        {
            amg_precond_iter(precond, v1, u1);
        } 
        else 
        {
            Eigen::VectorXd A_times_u1;
            matmul(u1, sparse_A, A_times_u1);
            Eigen::VectorXd r0 = v1 - A_times_u1;
            Eigen::VectorXd x1 = u1; 
            amg_precond_iter(precond, r0, x1);
            Eigen::VectorXd x2 = x1;
            dss_precond_iter(x1, v1, x2);

            for (int index = 0; index < x2.size(); ++index)
            {
                if (bad_indices_[0].count(index) == 0)
                {
                    x2(index) = v1(index) - sparse_A.row(index).dot(x1);
                }
                else 
                {
                    x2(index) -= x1(index);
                }
            }

            Eigen::VectorXd A_times_x2;
            matmul(x2, sparse_A, A_times_x2);
            Eigen::VectorXd r2 = v1 - A_times_x2;
            amg_precond_iter(precond, r2, x2);
            u1 = x2;
        }
        */
        
    
        eta = beta = sqrt(u1.dot(v1));
        gamma0 = gamma1 = 1.;
        sigma0 = sigma1 = 0.;

        norm_goal = conv_tol_;
        if (!use_absolute_tol)
        {
            norm_goal *= eta;
        }

        if (eta <= norm_goal)
        {
            num_iterations = 0;
            return;
        }

        for (num_iterations = 1; num_iterations <= max_iter_; ++num_iterations)
        {
            v1 /= beta;
            u1 /= beta;

            matmul(u1, sparse_A, q);
            alpha = u1.dot(q);

            if (num_iterations > 1)
            {
                q -= beta * v0;
            }

            v0 = q - alpha * v1;

            delta = gamma1 * alpha - gamma0 * sigma1 * beta;
            rho3 = sigma0 * beta;
            rho2 = sigma1 * alpha + gamma0 * gamma1 * beta;

            q.setZero();
            custom_mixed_precond_iter(precond, v0, q);
            /*
            if (!do_mixed_precond || bad_indices_.size() == 0 || bad_indices_[0].size() == 0)
            {
                amg_precond_iter(precond, v0, q);
            } 
            else 
            {
                Eigen::VectorXd A_times_q;
                matmul(q, sparse_A, A_times_q);
                Eigen::VectorXd r0 = v0 - A_times_q;
                Eigen::VectorXd x1 = q; 
                amg_precond_iter(precond, r0, x1);
                Eigen::VectorXd x2 = x1;
                dss_precond_iter(x1, v0, x2);

                for (int index = 0; index < x2.size(); ++index)
                {
                    if (bad_indices_[0].count(index) == 0)
                    {
                        x2(index) = v0(index) - sparse_A.row(index).dot(x1);
                    }
                    else 
                    {
                        x2(index) -= x1(index);
                    }
                }
                

                Eigen::VectorXd A_times_x2;
                matmul(x2, sparse_A, A_times_x2);
                Eigen::VectorXd r2 = v0 - A_times_x2;
                amg_precond_iter(precond, r2, x2);
                q = x2;
            }*/
            
            bool v0_isnan = false;
            bool q_isnan = false;
            for (int i = 0; i < q.size(); ++i)
            {
                if (std::isnan(v0(i)))
                {
                    v0_isnan = true;
                }
                if (std::isnan(q(i)))
                {
                    q_isnan = true;
                }
            }
            beta = sqrt(v0.dot(q));
            rho1 = std::hypot(delta, beta);
            logger->trace("DOT: {}", v0.dot(q));
            logger->trace("RHO1: {}", rho1);
            logger->trace("BETA: {}", beta);
            logger->trace("DELTA: {}", delta);
            logger->trace("qnan: {}", q_isnan);
            logger->trace("v0nan: {}", v0_isnan);

            if (v0_isnan)
            {
                std::ofstream A_file("fail_A.mat");
                A_file << sparse_A;
                A_file.close();

                std::ofstream b_file("fail_b.mat");
                b_file << rhs;
                b_file.close();
                exit(1);
            }

            if (num_iterations == 1)
            {
                w0 = u1 / rho1;
            }
            else if (num_iterations == 2)
            {
                w0 = 1 / rho1 * u1 - rho2 / rho1 * w1;
            }
            else
            {
                w0 = -rho3 / rho1 * w0 - rho2 / rho1 * w1;
                w0 += u1 / rho1;
            }

            gamma0 = gamma1;
            gamma1 = delta/rho1;

            result += gamma1 * eta * w0;

            sigma0 = sigma1;
            sigma1 = beta/rho1;

            eta = -sigma1 * eta;
            logger->trace("MINRES eta: {}, loop iter: {}", eta, num_iterations);
            Eigen::VectorXd A_times_x;
            matmul(result, sparse_A, A_times_x);
            Eigen::VectorXd r0 = rhs - A_times_x;
            logger->trace("RESIDUAL: {}", r0.norm());

            if (fabs(eta) <= norm_goal)
            {
                return;
            }

            Eigen::VectorXd temp;
            temp = u1;
            u1 = q;
            q = temp;
            
            temp = v0;
            v0 = v1;
            v1 = temp;

            temp = w0;
            w0 = w1;
            w1 = temp;
        }

    }

    void ExperimentalSolver::gmres_solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond)
    {
        double residual;
        int i, j = 1, k;
        Eigen::VectorXd s(m_ + 1);
        Eigen::VectorXd cs(m_ + 1);
        Eigen::VectorXd sn(m_ + 1);
        Eigen::VectorXd w(rhs.size());

        Eigen::VectorXd z(rhs.size());
        custom_mixed_precond_iter(precond, rhs, z);
        

        double normb = z.norm();
        Eigen::VectorXd A_times_x;
        matmul(result, sparse_A, A_times_x);
        Eigen::VectorXd r0 = rhs - A_times_x;

        Eigen::VectorXd r(rhs.size());
        custom_mixed_precond_iter(precond, r0, r);
        

        double beta = r.norm();
        if (normb == 0)
        {
            normb = 1;
        }

        residual = beta;
        if (!use_absolute_tol)
        {
            residual /= normb;
        }

        if (residual < conv_tol_)
        {
            num_iterations = 0;
            return;
        }

        Eigen::MatrixXd V(rhs.size(), m_ + 1);
        Eigen::MatrixXd H(m_ + 2, m_ + 1);
        H.setZero();

        while (j <= max_iter_)
        {
            V.col(0) = r / beta;
            s.setZero();
            s(0) = beta;

            for (i = 0; i < m_ && j <= max_iter_; ++i, ++j)
            {
                Eigen::VectorXd A_times_vi;
                Eigen::VectorXd vi = V.col(i);
                matmul(vi, sparse_A, A_times_vi);

                custom_mixed_precond_iter(precond, A_times_vi, w);
                

                for (k = 0; k <= i; ++k)
                {
                    H(k, i) = w.dot(V.col(k));
                    w -= H(k, i) * V.col(k);
                }

                H(i+1, i) = w.norm();
                V.col(i + 1) = w / H(i+1, i);
                
                for (k = 0; k < i; ++k)
                {
                    ApplyPlaneRotation(H(k,i), H(k+1,i), cs(k), sn(k));
                }
        
                GeneratePlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
                ApplyPlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
                ApplyPlaneRotation(s(i), s(i+1), cs(i), sn(i));
                
                residual = abs(s(i+1));
                if (!use_absolute_tol)
                {
                    residual /= normb;
                }

                logger->trace("GMRES. Iter: {}, Residual: {}", j, residual);

                if (residual < conv_tol_) 
                {
                    Update(result, i, H, s, V);
                    num_iterations = j;
                    return;
                }
            }

            Update(result, m_ - 1, H, s, V);
            matmul(result, sparse_A, A_times_x);
            r0 = rhs - A_times_x;

            r.setZero();
            custom_mixed_precond_iter(precond, r0, r);
            
            beta = r.norm();

            residual = beta;
            if (!use_absolute_tol)
            {
                residual /= normb;
            }

            if (residual < conv_tol_) 
            {
                num_iterations = j;
                return;
            }

        }
        num_iterations = max_iter_;
    }


    void ExperimentalSolver::custom_mixed_precond_iter(const HYPRE_Solver &precond, const Eigen::VectorXd &r, Eigen::VectorXd &z)
    {        
        Eigen::VectorXd z1(r.size());
        Eigen::VectorXd z2(r.size());
        Eigen::VectorXd z3(r.size());
        z1.setZero();
        z2.setZero();
        z3.setZero();

        assert(bad_indices_.size() == 1);
        if (!do_mixed_precond || bad_indices_.size() == 0 || bad_indices_[0].size() == 0)
        {
            amg_precond_iter(precond, r, z1);
            z = z1;
            return;
        }

        if (dss_in_middle)
        {
            amg_precond_iter(precond, r, z1);
            dss_precond_iter(z1, r, z2);
            Eigen::VectorXd A_times_z2;
            matmul(z2, sparse_A, A_times_z2);
            amg_precond_iter(precond, r - A_times_z2, z3);
            z = z2 + z3;
        }
        else
        {
            Eigen::VectorXd z0(r.size());
            z0.setZero();
            dss_precond_iter(z0, r, z1);
            Eigen::VectorXd A_times_z1;
            matmul(z2, sparse_A, A_times_z1);
            amg_precond_iter(precond, r - A_times_z1, z2);
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
        double solve_time;
        double copy_to_time;
        double copy_from_time;
#ifdef HYPRE_WITH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        {
            POLYSOLVE_SCOPED_STOPWATCH("copy to hypre time: ", copy_to_time, *logger);
            eigen_to_hypre_par_vec(par_x, x, eigen_x, start_i, end_i);
            eigen_to_hypre_par_vec(par_b, b, eigen_b, start_i, end_i);
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("boomeramg solve time: ", solve_time, *logger);
            HYPRE_BoomerAMGSolve(precond, parcsr_A, par_b, par_x);
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("copy from hypre time: ", copy_from_time, *logger);
            hypre_vec_to_eigen(x, eigen_x, start_i, end_i);
        }

        HYPRE_IJVectorDestroy(x);
        HYPRE_IJVectorDestroy(b);
        
    }

    void ExperimentalSolver::dss_precond_iter(const Eigen::VectorXd &z, const Eigen::VectorXd &r, Eigen::VectorXd &next_z)
    {
        double dss_step_time;
        {
            POLYSOLVE_SCOPED_STOPWATCH("dss step time: ", dss_step_time, *logger);

            next_z = z;
        
            #pragma omp parallel for
            for (int index = 0; index < bad_indices_.size(); ++index)
            {
                auto &subdomain = bad_indices_[index];

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

                sub_result = D_solvers[index].solve(sub_rhs);
                i_counter = 0;
                for (auto &i : subdomain)
                {
                    next_z(i) += sub_result(i_counter);
                    ++i_counter;
                }
            }
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

    void ExperimentalSolver::select_bad_indices(Eigen::VectorXd &rhs)
    {
        POLYSOLVE_SCOPED_STOPWATCH("bad dof selection time", bad_dof_selection_time, *logger);
        Eigen::VectorXd sq_mags(rhs.size());
        double cutoff_threshold; 

        if (!(select_bad_dofs_from_amg || select_bad_dofs_from_rhs || select_bad_dofs_from_row_norms))
        {
            return; 
        }

        if (select_bad_dofs_from_amg + select_bad_dofs_from_rhs + select_bad_dofs_from_row_norms > 1)
        {
            logger->warn("Multiple selection methods specified, defaulting to row norms.");   
        }

        bad_indices_.clear();
        bad_indices_.resize(1);

        if (select_bad_dofs_from_rhs)
        {
            double select_dofs_from_rhs_time;
            POLYSOLVE_SCOPED_STOPWATCH("select dofs from rhs", select_dofs_from_rhs_time, *logger);
            assert(rhs.size() % dimension_ == 0);
            for (int i = 0; i < rhs.size() / dimension_; ++i)
            {
                double sq_mag = rhs(dimension_ * i) * rhs(dimension_ * i);
                for (int j = 1; j < dimension_; ++j)
                {
                    sq_mag += rhs(dimension_ * i + j) * rhs(dimension_ * i + j);
                } 
                for (int j = 1; j < dimension_; ++j)
                {
                    sq_mags(dimension_ * i + j) = sq_mag;
                } 
            }

        }

        if (select_bad_dofs_from_amg)
        {
            double select_dofs_from_amg_time;
            POLYSOLVE_SCOPED_STOPWATCH("select dofs from amg", select_dofs_from_amg_time, *logger);
            HYPRE_ParVector test_par_b, test_par_x;
            HYPRE_IJVector test_x, test_b;

            Eigen::VectorXd test_result = Eigen::VectorXd::Random(rhs.size());
            Eigen::VectorXd start_result = test_result;
            Eigen::VectorXd test_rhs(rhs.size());
            test_rhs.setZero();

            eigen_to_hypre_par_vec(test_par_b, test_b, test_rhs, start_i, end_i);
            eigen_to_hypre_par_vec(test_par_x, test_x, test_result, start_i, end_i);

            HYPRE_Solver test_precond;
            HYPRE_BoomerAMGCreate(&test_precond);
            HypreBoomerAMG_SetDefaultOptions(test_precond);
            if (dimension_ > 1)
            {
                const int num_rbms = dimension_ == 2 ? 1 : 3;
                std::vector<HYPRE_ParVector> par_rbms(num_rbms);
                std::vector<HYPRE_IJVector> rbms(num_rbms);
                HypreBoomerAMG_SetElasticityOptions(
                    test_precond, 
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

            HYPRE_BoomerAMGSetMaxIter(test_precond, 5);
            HYPRE_BoomerAMGSetup(test_precond, parcsr_A, test_par_b, test_par_x);
            HYPRE_BoomerAMGSolve(test_precond, parcsr_A, test_par_b, test_par_x);

            hypre_vec_to_eigen(test_x, test_result, start_i, end_i);

            assert(rhs.size() % dimension_ == 0);
            Eigen::VectorXd sq_mags(rhs.size());
            for (int i = 0; i < rhs.size(); ++i)
            {
                sq_mags(i) = abs(test_result(i) / start_result(i));
            }
            
            HYPRE_IJVectorDestroy(test_x);
            HYPRE_IJVectorDestroy(test_b);
            HYPRE_BoomerAMGDestroy(test_precond);
        }

        if (select_bad_dofs_from_row_norms)
        {
            double select_dofs_from_row_norms_time;
            POLYSOLVE_SCOPED_STOPWATCH("select dofs from hess row norms", select_dofs_from_row_norms_time, *logger);
            assert(rhs.size() % dimension_ == 0);
            for (int i = 0; i < rhs.size(); ++i)
            {
                sq_mags(i) = sparse_A.row(i).norm();
            }
        }

        if (save_selection_criteria)
        {
            std::ofstream file;
            file.open("criteria.txt", std::ios_base::app);
            file << reduced_to_full_func(sq_mags).transpose() << std::endl;
            file.close();
        }

        Eigen::VectorXd sorted_sq_mags = sq_mags;
        std::sort(sorted_sq_mags.data(), sorted_sq_mags.data() + sorted_sq_mags.size());

        const int cutoff_index = sorted_sq_mags.size() * (1 - bad_dof_threshold);
        const double cutoff = sorted_sq_mags(cutoff_index);

        if (cutoff > 0)
        {
            for (int i = 0; i < rhs.size(); ++i)
            {
                if (sq_mags(i) >= cutoff)
                {
                    bad_indices_[0].insert(i);
                }
            }
        }

        if (save_selected_indices)
        {
            std::ofstream file;
            file.open("selected_indices.txt", std::ios_base::app);
            if (bad_indices_.size() > 0)
            {
                for (auto i : bad_indices_[0])
                {
                    file << i << " ";
                }
            }
            file << std::endl;;
            file.close();
        }

    }

    void ExperimentalSolver::factorize_submatrix()
    {
        {
            POLYSOLVE_SCOPED_STOPWATCH("assemble D", dss_assembly_time, *logger);
            D_solvers.clear();
            D_solvers.resize(bad_indices_.size());

            logger->trace("H symmetric: {}", sparse_A.isApprox(sparse_A.transpose()));

            #pragma omp parallel for
            for (int i = 0; i < bad_indices_.size(); ++i)
            {
                Eigen::SparseMatrix<double, Eigen::RowMajor> D;
                D.resize(bad_indices_[i].size(), bad_indices_[i].size());
                logger->trace("Subdomain size: {}", bad_indices_[i].size());
                std::vector<Eigen::Triplet<double>> triplets;
                std::unordered_map<int, int> index_mapping;

                int j_counter = 0;
                for (auto j : bad_indices_[i])
                {
                    index_mapping[j] = j_counter;
                    ++j_counter;
                }

                for (int k = 0; k < sparse_A.outerSize(); ++k)
                {
                    if (bad_indices_[i].count(k) == 0)
                    {
                        continue;
                    }
                    for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sparse_A, k); it; ++it)
                    {

                        if (bad_indices_[i].count(it.col()) == 0 || bad_indices_[i].count(it.row()) == 0)
                        {
                            continue;
                        }
                        triplets.push_back(Eigen::Triplet<double>(index_mapping[it.row()], index_mapping[it.col()], it.value()));
                    }
                }

                double set_from_triplets_time;
                {
                    POLYSOLVE_SCOPED_STOPWATCH("set D from triplets", set_from_triplets_time, *logger);
                    D.setFromTriplets(triplets.begin(), triplets.end());
                    logger->trace("D symmetric: {}", D.isApprox(D.transpose()));
                }

                {
                    POLYSOLVE_SCOPED_STOPWATCH("factorize D", dss_factorization_time, *logger);
                    D_solvers[i].compute(D);
                }
            

                // check symmetry
                int rows = D.rows();
                if (rows > 0)
                {
                    Eigen::VectorXd test_u(rows);
                    Eigen::VectorXd test_v(rows);
                    test_u.setRandom();
                    test_v.setRandom();
                    test_u /= test_u.norm();
                    test_v /= test_v.norm();
                    Eigen::VectorXd inverse_u = D_solvers[i].solve(test_u);
                    Eigen::VectorXd inverse_v = D_solvers[i].solve(test_v);
                    double sym_check = test_v.dot(inverse_u) - test_u.dot(inverse_v);
                    double sym_check2 = test_u.dot(inverse_u);
                    double sym_check3 = test_v.dot(inverse_v);
                    logger->trace("D Symmetry check: {}", sym_check);
                    logger->trace("D Symmetry check2: {}", sym_check2);
                    logger->trace("D Symmetry check3: {}", sym_check3);
                    logger->trace("Dmax: {}, Dmin: {}", D.coeffs().maxCoeff(), D.coeffs().minCoeff());
                }
            }
        }
    }

    void ExperimentalSolver::matmul(Eigen::VectorXd &x, Eigen::SparseMatrix<double, Eigen::RowMajor> &A, Eigen::VectorXd &result)
    {
#ifdef HYPRE_WITH_MPI
        result.resize(x.size());
        result.setZero();
        
        for (int i = start_i; i <= end_i; ++i)
        {
            result(i) = sparse_A.row(i).dot(x);
        }

        MPI_Allreduce(MPI_IN_PLACE, result.data(), result.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
        result = A*x;
#endif
    }

    void ExperimentalSolver::check_matrix_conditioning(const std::string name, const std::set<int>& subdomain)
    {
        if (subdomain.size() == 0)
        {
            return;
        }

        Eigen::MatrixXd preconditioned_A = sparse_A;

        for (int col = 0; col < preconditioned_A.cols(); ++col)
        {
            Eigen::VectorXd sub_rhs;
            Eigen::VectorXd sub_result;
            sub_rhs.resize(subdomain.size());
            sub_result.resize(subdomain.size());

            int i_counter = 0;
            for (auto &i : subdomain)
            {
                sub_rhs(i_counter) = preconditioned_A(i, col);
                ++i_counter;
            }

            sub_result = D_solvers[0].solve(sub_rhs);
            i_counter = 0;
            for (auto &i : subdomain)
            {
                preconditioned_A(i, col) = sub_result(i_counter);
                ++i_counter;
            }
        }

        check_matrix_conditioning(name, preconditioned_A);
    }

    void ExperimentalSolver::check_matrix_conditioning(const std::string name, const Eigen::MatrixXd& mat)
    {
        Eigen::BDCSVD<Eigen::MatrixXd> svd(mat);
        double cond = svd.singularValues()(0) 
        / svd.singularValues()(svd.singularValues().size()-1);

        Eigen::LDLT<Eigen::MatrixXd> chol_decomp(mat);
        bool spd = !(chol_decomp.info() == Eigen::NumericalIssue);
        bool symm = mat.isApprox(mat.transpose());
        bool isPos = chol_decomp.isPositive();

        logger->trace("Analyzing {}...", name);
        logger->trace("SPD: {}, Symm: {}, isPos: {}", spd, symm, isPos);
        logger->trace("Condition number: {}", cond);
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

