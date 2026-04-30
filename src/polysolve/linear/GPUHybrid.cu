
#include "GPUHybrid.hpp"

#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

#include <iostream>
#include <fstream>

#include <mpi.h>

#include <chrono>
#include <stdexcept>

#include <spdlog/spdlog.h>


#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorName(status) \
                      << " (" << cudaGetErrorString(status) << ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUDSS(call) \
    do { \
        cudssStatus_t status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            std::cerr << "cuDSS Error at " << __FILE__ << ":" << __LINE__ \
                      << " code " << (int) status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


namespace polysolve::linear
{

    namespace
    {
        using clock = std::chrono::steady_clock;

        double elapsed_seconds(const std::chrono::time_point<clock> &begin)
        {
            return std::chrono::duration<double>(clock::now() - begin).count();
        }
    }

    GPUHybrid::GPUHybrid()
    {
        HYPRE_Init();
        CHECK_CUDA(cudaSetDevice(0));
        HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
        HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
        HYPRE_SetSpGemmUseCusparse(false);
        HYPRE_SetUseGpuRand(true);

        CHECK_CUDSS(cudssCreate(&cudss_handle));
    }

    void GPUHybrid::set_parameters(const json &params)
    {
        if (params.contains("GPUHybrid"))
        {
            if (params["GPUHybrid"].contains("max_iter"))
            {
                max_iter_ = params["GPUHybrid"]["max_iter"];
            }
            if (params["GPUHybrid"].contains("tolerance"))
            {
                conv_tol_ = params["GPUHybrid"]["tolerance"];
            }
            if (params["GPUHybrid"].contains("theta"))
            {
                theta = params["GPUHybrid"]["theta"];
            }
            if (params["GPUHybrid"].contains("dimension"))
            {
                dimension_ = params["GPUHybrid"]["dimension"];
            }
            if (params["GPUHybrid"].contains("do_mixed_precond"))
            {
                do_mixed_precond = params["GPUHybrid"]["do_mixed_precond"];
            }
            if (params["GPUHybrid"].contains("use_absolute_tol"))
            {
                use_absolute_tol = params["GPUHybrid"]["use_absolute_tol"];
            }
            if (params["GPUHybrid"].contains("decompose_subdomains"))
            {
                decompose_subdomains = params["GPUHybrid"]["decompose_subdomains"];
            }
            if (params["GPUHybrid"].contains("min_subdomain_size"))
            {
                min_subdomain_size = params["GPUHybrid"]["min_subdomain_size"];
            }
            if (params["GPUHybrid"].contains("max_subdomain_size"))
            {
                max_subdomain_size = params["GPUHybrid"]["max_subdomain_size"];
            }
            if (params["GPUHybrid"].contains("bad_dof_threshold"))
            {
                bad_dof_threshold = params["GPUHybrid"]["bad_dof_threshold"];
            }
            if (params["GPUHybrid"].contains("max_dense_size"))
            {
                max_dense_size = params["GPUHybrid"]["max_dense_size"];
            }   
        }
    } 

    void GPUHybrid::get_info(json &params) const
    {
        params["num_iterations"] = num_iterations;
        params["final_res_norm"] = final_res_norm;
    }

    void GPUHybrid::check_settings() const
    {

    }

    void GPUHybrid::analyze_pattern(const StiffnessMatrix &A, const int precond_num)
    {
        check_settings();
    }

    void GPUHybrid::factorize(const StiffnessMatrix &Ain)
    {
        {
            POLYSOLVE_SCOPED_STOPWATCH("eigen matrix copy time", eigen_copy_time, *logger);
            sparse_A = Ain;
            sparse_A.makeCompressed();
        }

        if (do_mixed_precond)
        {
            prepare_dss();
        }

        if (has_matrix_)
        {
            POLYSOLVE_SCOPED_STOPWATCH("matrix destroy time", matrix_destroy_time, *logger);
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }

        copy_matrix_to_hypre();
    }

    namespace {
        void HypreBoomerAMG_SetDefaultOptions(HYPRE_Solver &amg_precond)
        {
            // AMG coarsening options:
            int coarsen_type = 8; // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
            int agg_levels = 1;    // number of aggressive coarsening levels
            double theta = 0.25;   // strength threshold: 0.25, 0.5, 0.8

            // AMG interpolation options:
            int interp_type = 6; // 6 = extended+i, 0 = classical
            int Pmax = 4;        // max number of elements per row in P

            // AMG relaxation options:
            int relax_type = 18;   // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
            int relax_sweeps = 1; // relaxation sweeps on each level

            // Additional options:
            int print_level = 0; // print AMG iterations? 1 = no, 2 = yes
            int max_levels = 25; // max number of levels in AMG hierarchy

            int min_coarse_size = 5;

            HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
            HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);

            HYPRE_BoomerAMGSetRelaxOrder(amg_precond, false);
            HYPRE_BoomerAMGSetRAP2(amg_precond, true);
            HYPRE_BoomerAMGSetKeepTranspose(amg_precond, true);
            
            HYPRE_BoomerAMGSetMinCoarseSize(amg_precond, min_coarse_size);
            HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_type, 3);
            HYPRE_BoomerAMGSetNumSweeps(amg_precond, relax_sweeps);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
            HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type);
            HYPRE_BoomerAMGSetPMaxElmts(amg_precond, Pmax);
            //print_level = 3;
            HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level);
            HYPRE_BoomerAMGSetMaxLevels(amg_precond, max_levels);

            // Use as a preconditioner (one V-cycle, zero tolerance)
            HYPRE_BoomerAMGSetMaxIter(amg_precond, 1);
            HYPRE_BoomerAMGSetTol(amg_precond, 0.0);
        }

        void HypreBoomerAMG_SetElasticityOptions(HYPRE_Solver &amg_precond, int dim, double theta)
        {
            // Make sure the systems AMG options are set
            HYPRE_BoomerAMGSetNumFunctions(amg_precond, dim);

            //HYPRE_BoomerAMGSetDofFunc(amg_precond, (HYPRE_Int*) dof_to_function.data());

            // More robust options with respect to convergence
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
        }
    }

    void GPUHybrid::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        Eigen::VectorXd result = x;
        Eigen::VectorXd rhs = b;

        HYPRE_ParVector par_b;
        HYPRE_ParVector par_x;
        init_hypre_vectors(b.size());            
        
        double *gpu_rhs;
        double *gpu_x;
        CHECK_CUDA(cudaMalloc(&gpu_rhs, rhs.size() * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&gpu_x, x.size() * sizeof(double)));
        CHECK_CUDA(cudaMemcpy(gpu_rhs, b.data(), rhs.size() * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(gpu_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice));

        set_hypre_vec(ij_b, par_b, gpu_rhs);
        set_hypre_vec(ij_x, par_x, gpu_x);
        
        HYPRE_Solver precond;

        HYPRE_BoomerAMGCreate(&precond);

        {
            POLYSOLVE_SCOPED_STOPWATCH("set AMG options", set_options_time, *logger);
            HypreBoomerAMG_SetDefaultOptions(precond);
            if (dimension_ > 1)
            {
                HypreBoomerAMG_SetElasticityOptions(
                    precond, 
                    dimension_, 
                    theta
                );
            }
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("AMG setup time", amg_setup_time, *logger);
            HYPRE_BoomerAMGSetup(precond, parcsr_A, par_b, par_x);
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("actual solve time", actual_solve_time, *logger);

            pcg_solve(gpu_rhs, gpu_x, par_b, par_x, precond);
        }

        CHECK_CUDA(cudaMemcpy(x.data(), gpu_x, x.size() * sizeof(double), cudaMemcpyDeviceToHost));
        final_res_norm = (b - sparse_A * x).norm();

        logger->debug("GPUHybrid solver Iterations: {}", num_iterations);
        logger->debug("GPUHybrid solver Final Relative Residual Norm: {}", final_res_norm);

        {
            POLYSOLVE_SCOPED_STOPWATCH("destroy time", destroy_time, *logger);
            HYPRE_BoomerAMGDestroy(precond);
            HYPRE_IJVectorDestroy(ij_x);
            HYPRE_IJVectorDestroy(ij_b);

            CHECK_CUDA(cudaFree(gpu_x));
            CHECK_CUDA(cudaFree(gpu_rhs));
        }
    }

    void GPUHybrid::copy_matrix_to_hypre()
    {
        POLYSOLVE_SCOPED_STOPWATCH("copy matrix time", matrix_copy_time, *logger);

        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, sparse_A.rows() - 1, 0, sparse_A.cols() - 1, &A);
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A);

        HYPRE_Int num_rows = sparse_A.rows();
        HYPRE_Int nnz = sparse_A.nonZeros();

        std::vector<HYPRE_Int> cpu_n_cols(num_rows, 0);
        std::vector<HYPRE_Int> cpu_rows(num_rows, 0);
        std::vector<HYPRE_Int> cpu_cols;
        std::vector<double> cpu_vals;

        cpu_cols.reserve(nnz);
        cpu_vals.reserve(nnz);

        // assuming symmetry
        for (HYPRE_Int k = 0; k < num_rows; ++k)
        {
            cpu_rows[k] = k;
            int counter = 0;
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sparse_A, k); it; ++it)
            {
                cpu_cols.push_back((HYPRE_Int)it.col());
                cpu_vals.push_back(it.value());
                counter++;
            }
            cpu_n_cols[k] = counter;
        }

        HYPRE_Int *gpu_n_cols, *gpu_rows, *gpu_cols;
        double *gpu_vals;

        CHECK_CUDA(cudaMalloc(&gpu_n_cols, num_rows * sizeof(HYPRE_Int)));
        CHECK_CUDA(cudaMalloc(&gpu_rows, num_rows * sizeof(HYPRE_Int)));
        CHECK_CUDA(cudaMalloc(&gpu_cols, nnz * sizeof(HYPRE_Int)));
        CHECK_CUDA(cudaMalloc(&gpu_vals, nnz * sizeof(double)));

        CHECK_CUDA(cudaMemcpy(gpu_n_cols, cpu_n_cols.data(), num_rows * sizeof(HYPRE_Int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(gpu_rows, cpu_rows.data(), num_rows * sizeof(HYPRE_Int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(gpu_cols, cpu_cols.data(), nnz * sizeof(HYPRE_Int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(gpu_vals, cpu_vals.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));

        HYPRE_IJMatrixSetValues(A, num_rows, gpu_n_cols, gpu_rows, gpu_cols, gpu_vals);

        CHECK_CUDA(cudaFree(gpu_n_cols));
        CHECK_CUDA(cudaFree(gpu_rows));
        CHECK_CUDA(cudaFree(gpu_cols));
        CHECK_CUDA(cudaFree(gpu_vals));

        HYPRE_IJMatrixAssemble(A);

        void* temp_A = nullptr;
        HYPRE_IJMatrixGetObject(A, &temp_A);
        parcsr_A = static_cast<decltype(parcsr_A)>(temp_A);
    }

    void GPUHybrid::init_hypre_vectors(const int size)
    {
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, size - 1, &ij_x);
        HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
        HYPRE_IJVectorInitializeShell(ij_x);
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, size - 1, &ij_b);
        HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
        HYPRE_IJVectorInitializeShell(ij_b);
    }

    void GPUHybrid::matmul(double* x, double* result)
    {
        POLYSOLVE_SCOPED_STOPWATCH("matmul time", matmul_time, *logger);     
        
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_result;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_result, result);

        HYPRE_ParCSRMatrixMatvec(1.0, parcsr_A, par_x, 0.0, par_result);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    double GPUHybrid::dot(double* a, double* b)
    {
        HYPRE_ParVector par_a;
        HYPRE_ParVector par_b;

        set_hypre_vec(ij_x, par_a, a);
        set_hypre_vec(ij_b, par_b, b);
        double result;
        HYPRE_ParVectorInnerProd(par_a, par_b, &result);
        return result;
    }

    void GPUHybrid::vector_copy(double* x, double* y)
    {
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_y;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_y, y);

        HYPRE_ParVectorCopy(par_x, par_y);
    }

    void GPUHybrid::vector_add(double alpha, double* x, double* y)
    {
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_y;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_y, y);

        hypre_ParVectorAxpy(alpha, par_x, par_y);
    }

    void GPUHybrid::vector_scale(double alpha, double* x)
    {
        HYPRE_ParVector par_x;

        set_hypre_vec(ij_x, par_x, x);

        HYPRE_ParVectorScale(alpha, par_x);
    }

    void GPUHybrid::set_hypre_vec(HYPRE_IJVector &my_ij_x, HYPRE_ParVector &par_x, double* x)
    {
        HYPRE_IJVectorSetData(my_ij_x, x);
        HYPRE_IJVectorAssemble(my_ij_x);
        HYPRE_IJVectorGetObject(my_ij_x, (void **)&par_x);
    }

    void GPUHybrid::custom_mixed_precond_iter(const HYPRE_Solver &precond, double* r, double* z, double* buffer, double* z2)
    {
        assert(bad_indices_.size() == 1);
        if (!do_mixed_precond || bad_indices_.size() == 0 || bad_indices_[0].size() == 0)
        {
            amg_precond_iter(precond, r, z);
            return;
        }
        else
        {
            CHECK_CUDA(cudaMemset(buffer, 0, sparse_A.rows() * sizeof(double)));
            CHECK_CUDA(cudaMemset(z2, 0, sparse_A.rows() * sizeof(double)));
            amg_precond_iter(precond, r, z);
            dss_precond_iter(z, r, z2);
            matmul(z2, z);
            vector_copy(r, buffer);
            vector_add(-1.0, z, buffer);
            CHECK_CUDA(cudaMemset(z, 0, sparse_A.rows() * sizeof(double)));
            amg_precond_iter(precond, buffer, z);
            vector_add(1.0, z2, z);
        }

    }

    void GPUHybrid::dss_precond_iter(double* z, double* r, double* next_z)
    {
       {
            POLYSOLVE_SCOPED_STOPWATCH("dss step time: ", dss_step_time, *logger);

            CHECK_CUDA(cudaMemset(next_z, 0, sparse_A.rows() * sizeof(double)));

            matmul(z, next_z);
            vector_scale(-1.0, next_z);
            vector_add(1.0, r, next_z);

            thrust::gather(
                thrust::device,
                all_bad_dof_map.begin(), 
                all_bad_dof_map.end(), 
                next_z, 
                d_b
            );
            
            CHECK_CUDA(cudaDeviceSynchronize());
            double test_time;
            {
                POLYSOLVE_SCOPED_STOPWATCH("dss backsub time: ", test_time, *logger);

                {
                    ///CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE, config, solverData, 
                    //                        batchMatrixA, batchMatrixX, batchMatrixB));
                    
                    // ==========================================
                    //            DENSE SOLVE (cuBLAS)
                    // ==========================================
                    if (dense_batch_count > 0) {

                        // 1. Zero out the padded RHS space to ensure empty padding solves to 0 safely
                        CHECK_CUDA(cudaMemset(d_dense_x, 0, dense_batch_count * max_dense_dim * sizeof(double)));

                        // 2. Scatter B into dense padded RHS vectors
                        void** d_b_ptrs_device = d_b_void; 

                        // Create local copies of class members to safely capture in __device__ lambdas
                        int* local_d_orig_idx = d_d_orig_idx;
                        int* local_d_batch_id = d_d_batch_id;
                        int* local_d_local_offset = d_d_local_offset;
                        int* local_d_nrows = d_d_nrows;
                        double* local_d_dense_x = d_dense_x;
                        int local_max_dense_dim = max_dense_dim;

                        thrust::for_each(thrust::device,
                            thrust::make_counting_iterator<int>(0),
                            thrust::make_counting_iterator<int>(num_dense_subsystems),
                            [=] __device__ (int tid) {
                                int orig_idx = local_d_orig_idx[tid];
                                int batch_id = local_d_batch_id[tid];
                                int offset   = local_d_local_offset[tid];
                                int nrows    = local_d_nrows[tid];

                                double* b_src = static_cast<double*>(d_b_ptrs_device[orig_idx]);
                                double* b_dst = local_d_dense_x + (batch_id * local_max_dense_dim) + offset;

                                for (int i = 0; i < nrows; ++i) {
                                    b_dst[i] = b_src[i];
                                }
                            }
                        );
                        CHECK_CUDA(cudaDeviceSynchronize());

                        int info_solve = 0;
                        
                        // 3. getrsBatched solves in-place, meaning d_dense_x_ptrs will be overwritten with the solution X
                        int err_check = cublasDgetrsBatched(
                            cublas_handle,
                            CUBLAS_OP_N,
                            max_dense_dim,
                            1,              // nrhs (number of right hand sides per system)
                            d_dense_ptrs,   // LU factored matrices generated during factorization
                            max_dense_dim,
                            d_pivots,       // Pivots generated during factorization
                            d_dense_x_ptrs, // RHS arrays
                            max_dense_dim,
                            &info_solve,
                            dense_batch_count
                        ); 

                        // 4. Gather Solution X back to the original pointers
                        void** d_x_ptrs_device = d_x_void; 

                        thrust::for_each(thrust::device,
                            thrust::make_counting_iterator<int>(0),
                            thrust::make_counting_iterator<int>(num_dense_subsystems),
                            [=] __device__ (int tid) {
                                int orig_idx = local_d_orig_idx[tid];
                                int batch_id = local_d_batch_id[tid];
                                int offset   = local_d_local_offset[tid];
                                int nrows    = local_d_nrows[tid];

                                double* x_src = local_d_dense_x + (batch_id * local_max_dense_dim) + offset;
                                double* x_dst = static_cast<double*>(d_x_ptrs_device[orig_idx]);

                                for (int i = 0; i < nrows; ++i) {
                                    x_dst[i] = x_src[i];
                                }
                            }
                        );
                    }

                    // ==========================================
                    //            SPARSE SOLVE (cuDSS)
                    // ==========================================
                    if (sparseBatchCount > 0) {                     
                        CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE, config, solverData, 
                                                batchMatrixA, batchMatrixX, batchMatrixB));
                        
                    }
                    CHECK_CUDA(cudaDeviceSynchronize());
                }
            }
            
            CHECK_CUDA(cudaMemset(next_z, 0, sizeof(double) * sparse_A.rows()));
    
            thrust::scatter(
                thrust::device,
                d_x,
                d_x + all_bad_dof_map.size(),
                all_bad_dof_map.begin(),
                next_z
            );

            vector_add(1.0, z, next_z);
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    void GPUHybrid::amg_precond_iter(const HYPRE_Solver &precond, double* b, double* x)
    {
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_b;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_b, b);

        {
            POLYSOLVE_SCOPED_STOPWATCH("boomeramg solve time: ", solve_time, *logger);
            HYPRE_BoomerAMGSolve(precond, parcsr_A, par_b, par_x);
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    void GPUHybrid::prepare_dss()
    {
        POLYSOLVE_SCOPED_STOPWATCH("prepare dss time", prepare_dss_time, *logger);

        select_bad_indices();
        std::vector<std::set<int>> overlap_extensions;

        if (decompose_subdomains)
        {
            decompose_subdomains_to_disjoint_subsets(overlap_extensions);
        }

        bad_indices_arrays.clear();
        logger->trace("Num subdomains: {}", bad_indices_.size());

        int i = 0;
        for (auto& subdomain : bad_indices_)
        {
            std::vector<int> cpu_buff(subdomain.begin(), subdomain.end());
            bad_indices_arrays.emplace_back(cpu_buff.begin(), cpu_buff.end());
            ++i;
        }

        CHECK_CUDA(cudaDeviceSynchronize());
        factorize_submatrix();
    }

    void GPUHybrid::decompose_subdomains_to_disjoint_subsets(std::vector<std::set<int>> &overlap_extensions)
    {
        POLYSOLVE_SCOPED_STOPWATCH("subdomain decomposition time", decomp_time, *logger);
        std::vector<int> all_bad_dofs;
        std::vector<int> global_to_local(sparse_A.rows(), -1);
        for (auto &subdomain : bad_indices_)
        {
            for (auto index : subdomain)
            {
                global_to_local[index] = all_bad_dofs.size();
                all_bad_dofs.push_back(index);
            }
        }

        disjointSet decomposed_subdomains(all_bad_dofs.size());

        for (int k : all_bad_dofs)
        {
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sparse_A, k); it; ++it)
            {
                if (global_to_local[it.col()] != -1)
                {
                    decomposed_subdomains.union_set(global_to_local[it.row()], global_to_local[it.col()]);
                }
            }
        }

        std::unordered_map<int, std::vector<int>> chosen_sets;
        for (auto index : all_bad_dofs)
        {
            chosen_sets[decomposed_subdomains.find_set(global_to_local[index])].push_back(index);
        }

        bad_indices_.clear();

        for (auto &kv : chosen_sets)
        {
            if (kv.second.size() < min_subdomain_size)
            {
                continue;
            }
            if (kv.second.size() > max_subdomain_size)
            {
                continue;
            }
            bad_indices_.emplace_back(kv.second.begin(), kv.second.end());
            overlap_extensions.emplace_back();
        }
    }

    void GPUHybrid::select_bad_indices()
    {
        POLYSOLVE_SCOPED_STOPWATCH("bad dof selection time", bad_dof_selection_time, *logger);
        
        bad_indices_.clear();
        bad_indices_.resize(1);
        Eigen::VectorXd sq_mags;
        {
            POLYSOLVE_SCOPED_STOPWATCH("select dofs from hess diagonal", select_dofs_from_diag_time, *logger);
            sq_mags = sparse_A.diagonal().cwiseAbs();
        }

        int n = sq_mags.size();
        if (n == 0) return; 

        thrust::device_vector<double> d_sq_mags(sq_mags.data(), sq_mags.data() + n);

        thrust::device_vector<int> d_indices(n);
        thrust::sequence(d_indices.begin(), d_indices.end());

        thrust::sort_by_key(d_sq_mags.begin(), d_sq_mags.end(), d_indices.begin());

        thrust::device_vector<double> d_log_sorted(n);
        thrust::transform(d_sq_mags.begin(), d_sq_mags.end(), d_log_sorted.begin(), 
                          [] __device__ (double val) { return ::log(val); });

        double min_mag = d_log_sorted.front();
        double max_mag = d_log_sorted.back();

        thrust::device_vector<double> d_deviations(n);
        
        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(n),
            d_log_sorted.begin(),
            d_deviations.begin(),
            [min_mag, max_mag, n] __device__ (int i, double log_y) {
                double expected_y = (max_mag - min_mag) / (n - 1.0) * i + min_mag;
                return fabs(expected_y - log_y);
            }
        );

        auto max_dev_iter = thrust::max_element(d_deviations.begin(), d_deviations.end());
        int cutoff_index = thrust::distance(d_deviations.begin(), max_dev_iter);
        
        if (bad_dof_threshold < 1.0)
        {
            cutoff_index = n * (1.0 - bad_dof_threshold);
        }

        const double cutoff = d_sq_mags[cutoff_index];
        logger->trace("Problematic threshold: {}, cutoff index: {}", cutoff, cutoff_index);

        int num_bad = n - cutoff_index;
        std::vector<int> h_bad_indices(num_bad);
        thrust::copy(d_indices.begin() + cutoff_index, d_indices.end(), h_bad_indices.begin());

        for (int idx : h_bad_indices)
        {
            bad_indices_[0].insert(idx);
        }
    }

    void GPUHybrid::allocate_subdomains()
    {
        int batchCount = bad_indices_arrays.size();

        free_device_memory();
        
        CHECK_CUDSS(cudssConfigCreate(&config));
        CHECK_CUDSS(cudssDataCreate(cudss_handle, &solverData));

        h_nrows.resize(batchCount);
        h_ncols.resize(batchCount);
        h_nnz.resize(batchCount);
        h_vec_ncols.resize(batchCount, 1);
        h_ld.resize(batchCount);

        h_csrRowOffsets_void.resize(batchCount);
        h_csrColIndices_void.resize(batchCount);
        h_csrValues_void.resize(batchCount);

        row_starts.resize(batchCount + 1);
        nnz_starts.resize(batchCount);
        dof_starts.resize(batchCount + 1);

        std::vector<int> h_bad_offsets(batchCount + 1, 0);
        all_bad_dof_map.clear();
        int total_bad_dofs = 0;

        for (int i = 0; i < batchCount; ++i) 
        {
            auto& ba = bad_indices_arrays[i];
            all_bad_dof_map.insert(all_bad_dof_map.end(), ba.begin(), ba.end());
            
            h_nrows[i] = ba.size();
            h_ncols[i] = ba.size();
            h_ld[i]    = ba.size();
            
            total_bad_dofs += ba.size();
            h_bad_offsets[i + 1] = total_bad_dofs;
        }
        logger->trace("Total bad dofs: {}", total_bad_dofs);

        int* d_outer_ptrs;
        int* d_inner_indices;
        double* d_values;
        int sparse_A_rows = sparse_A.rows();
        int sparse_A_nnz = sparse_A.nonZeros();
        
        CHECK_CUDA(cudaMalloc(&d_outer_ptrs, (sparse_A_rows + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_inner_indices, sparse_A_nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_values, sparse_A_nnz * sizeof(double)));
        
        CHECK_CUDA(cudaMemcpy(d_outer_ptrs, sparse_A.outerIndexPtr(), (sparse_A_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_inner_indices, sparse_A.innerIndexPtr(), sparse_A_nnz * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_values, sparse_A.valuePtr(), sparse_A_nnz * sizeof(double), cudaMemcpyHostToDevice));

        // --- ALLOCATE TEMPORARY DEVICE METADATA ---
        int* d_bad_indices;
        int* d_bad_offsets;
        int* d_nnz_per_row;
        int* d_value_offsets;

        CHECK_CUDA(cudaMalloc(&d_bad_indices, total_bad_dofs * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_bad_offsets, (batchCount + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_nnz_per_row, total_bad_dofs * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_value_offsets, (total_bad_dofs + 1) * sizeof(int)));

        CHECK_CUDA(cudaMemcpy(d_bad_indices, all_bad_dof_map.data().get(), total_bad_dofs * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_bad_offsets, h_bad_offsets.data(), (batchCount + 1) * sizeof(int), cudaMemcpyHostToDevice));

        // --- PASS 1: COUNT NNZ PER ROW ---
        thrust::for_each(thrust::device,
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(total_bad_dofs),
            [=] __device__ (int tid) {
                int l = 0, r = batchCount - 1;
                int batch_id = 0;
                while (l <= r) {
                    int m = l + (r - l) / 2;
                    if (d_bad_offsets[m] <= tid && tid < d_bad_offsets[m + 1]) {
                        batch_id = m; break;
                    } else if (d_bad_offsets[m] > tid) r = m - 1;
                    else l = m + 1;
                }

                int batch_start = d_bad_offsets[batch_id];
                int batch_end = d_bad_offsets[batch_id + 1];
                int global_row = d_bad_indices[tid];

                int ptr_start = d_outer_ptrs[global_row];
                int ptr_end = d_outer_ptrs[global_row + 1];

                int count = 0;
                for (int p = ptr_start; p < ptr_end; ++p) {
                    int g_col = d_inner_indices[p];
                    int cl = batch_start, cr = batch_end - 1;
                    while (cl <= cr) {
                        int cm = cl + (cr - cl) / 2;
                        if (d_bad_indices[cm] == g_col) { count++; break; }
                        if (d_bad_indices[cm] < g_col) cl = cm + 1;
                        else cr = cm - 1;
                    }
                }
                d_nnz_per_row[tid] = count;
            }
        );

        // Prefix sum (exclusive scan)
        thrust::device_ptr<int> t_nnz_per_row(d_nnz_per_row);
        thrust::device_ptr<int> t_value_offsets(d_value_offsets);
        thrust::exclusive_scan(thrust::device, t_nnz_per_row, t_nnz_per_row + total_bad_dofs + 1, t_value_offsets);

        // --- FETCH METADATA AND COMPUTE OFFSETS ---
        std::vector<int> h_value_offsets(batchCount + 1);
        for(int i = 0; i <= batchCount; ++i) {
            CHECK_CUDA(cudaMemcpy(&h_value_offsets[i], d_value_offsets + h_bad_offsets[i], sizeof(int), cudaMemcpyDeviceToHost));
        }

        int total_nnz = h_value_offsets[batchCount];
        int total_row_offsets = total_bad_dofs + batchCount;

        int running_row_offsets = 0;
        int running_dofs = 0;
        for (int i = 0; i < batchCount; ++i) {
            row_starts[i] = running_row_offsets;
            nnz_starts[i] = h_value_offsets[i]; 
            dof_starts[i] = running_dofs;
            
            h_nnz[i] = h_value_offsets[i+1] - h_value_offsets[i];
            
            running_row_offsets += h_nrows[i] + 1;
            running_dofs += h_nrows[i];
            logger->trace("Subdomain size: {}, NNZ: {}, Fill: {}", h_nrows[i], h_nnz[i], static_cast<double>(h_nnz[i]) / (h_nrows[i] * h_nrows[i]));
        }
        row_starts[batchCount] = running_row_offsets;
        dof_starts[batchCount] = running_dofs;

        CHECK_CUDA(cudaMalloc(&d_all_rowOffsets, total_row_offsets * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_all_colIndices, total_nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_all_values, total_nnz * sizeof(double)));

        thrust::for_each(thrust::device,
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(total_bad_dofs),
            [=] __device__ (int tid) {
                int l = 0, r = batchCount - 1;
                int batch_id = 0;
                while (l <= r) {
                    int m = l + (r - l) / 2;
                    if (d_bad_offsets[m] <= tid && tid < d_bad_offsets[m + 1]) {
                        batch_id = m; break;
                    } else if (d_bad_offsets[m] > tid) r = m - 1;
                    else l = m + 1;
                }

                int batch_start = d_bad_offsets[batch_id];
                int batch_end = d_bad_offsets[batch_id + 1];
                int local_row = tid - batch_start;
                int global_row = d_bad_indices[tid];

                int write_ptr = d_value_offsets[tid];
                int batch_nnz_start = d_value_offsets[batch_start];
                
                d_all_rowOffsets[batch_id + tid] = write_ptr - batch_nnz_start;

                if (local_row == (batch_end - batch_start - 1)) {
                    d_all_rowOffsets[batch_id + tid + 1] = d_value_offsets[batch_end] - batch_nnz_start;
                }

                int ptr_start = d_outer_ptrs[global_row];
                int ptr_end = d_outer_ptrs[global_row + 1];

                for (int p = ptr_start; p < ptr_end; ++p) {
                    int g_col = d_inner_indices[p];
                    int cl = batch_start, cr = batch_end - 1;
                    while (cl <= cr) {
                        int cm = cl + (cr - cl) / 2;
                        if (d_bad_indices[cm] == g_col) {
                            d_all_colIndices[write_ptr] = cm - batch_start; 
                            d_all_values[write_ptr] = d_values[p];
                            write_ptr++;
                            break;
                        }
                        if (d_bad_indices[cm] < g_col) cl = cm + 1;
                        else cr = cm - 1;
                    }
                }
            }
        );

        // --- LINK UP CUDSS POINTERS ---
        for (int i = 0; i < batchCount; ++i)
        {
            h_csrRowOffsets_void[i] = static_cast<void*>(d_all_rowOffsets + row_starts[i]);
            h_csrColIndices_void[i] = static_cast<void*>(d_all_colIndices + nnz_starts[i]);
            h_csrValues_void[i] = static_cast<void*>(d_all_values + nnz_starts[i]);
        }

        CHECK_CUDA(cudaMalloc(&d_csrRowOffsets_void, batchCount * sizeof(void*)));
        CHECK_CUDA(cudaMalloc(&d_csrColIndices_void, batchCount * sizeof(void*)));
        CHECK_CUDA(cudaMalloc(&d_csrValues_void, batchCount * sizeof(void*)));

        CHECK_CUDA(cudaMalloc(&d_x, total_bad_dofs * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_b, total_bad_dofs * sizeof(double)));

        h_x_void.clear();
        h_b_void.clear();

        for (int i = 0; i < batchCount; ++i)
        {
            h_x_void.push_back(static_cast<void*>(d_x + dof_starts[i]));
            h_b_void.push_back(static_cast<void*>(d_b + dof_starts[i]));
        }

        CHECK_CUDA(cudaMalloc(&d_x_void, batchCount * sizeof(void*)));
        CHECK_CUDA(cudaMalloc(&d_b_void, batchCount * sizeof(void*)));

        CHECK_CUDA(cudaMemcpy(d_csrRowOffsets_void, h_csrRowOffsets_void.data(), batchCount * sizeof(void*), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_csrColIndices_void, h_csrColIndices_void.data(), batchCount * sizeof(void*), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_csrValues_void, h_csrValues_void.data(), batchCount * sizeof(void*), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMemcpy(d_x_void, h_x_void.data(), batchCount * sizeof(void*), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b_void, h_b_void.data(), batchCount * sizeof(void*), cudaMemcpyHostToDevice));

        cudaFree(d_bad_indices);
        cudaFree(d_bad_offsets);
        cudaFree(d_nnz_per_row);
        cudaFree(d_value_offsets);
        cudaFree(d_outer_ptrs);
        cudaFree(d_inner_indices);
        cudaFree(d_values);
    }

    void GPUHybrid::factorize_submatrix()
    {
        POLYSOLVE_SCOPED_STOPWATCH("assemble D", dss_assembly_time, *logger);

        int batchCount = bad_indices_arrays.size();
        if (batchCount == 0) return;

        allocate_subdomains();

        {
            std::vector<int> dense_indices;
            std::vector<int> sparse_indices;
            max_dense_dim = 0;

            for (int i = 0; i < batchCount; ++i) {
                if (h_nrows[i] <= max_dense_size) {
                    dense_indices.push_back(i);
                    max_dense_dim = std::max(max_dense_dim, h_nrows[i]);
                } else {
                    sparse_indices.push_back(i);
                }
            }

            // ==========================================
            //            DENSE PROCESSING
            // ==========================================
            if (!dense_indices.empty()) {
                // Pack Dense Subsystems
                struct PackedBatch {
                    std::vector<int> orig_indices;
                    int used_size = 0;
                };
                std::vector<PackedBatch> packed_batches;

                for (int idx : dense_indices) {
                    int size = h_nrows[idx];
                    bool placed = false;
                    for (auto& pb : packed_batches) {
                        if (pb.used_size + size <= max_dense_dim) {
                            pb.orig_indices.push_back(idx);
                            pb.used_size += size;
                            placed = true;
                            break;
                        }
                    }
                    if (!placed) {
                        packed_batches.push_back({{idx}, size});
                    }
                }
    
                // Assign to class member
                dense_batch_count = packed_batches.size();
                
                // Metadata for mapping preassembled subsystems to packed batches
                std::vector<int> h_d_orig_idx;
                std::vector<int> h_d_batch_id;
                std::vector<int> h_d_local_offset;
                std::vector<int> h_d_nrows;
                
                for (int b = 0; b < dense_batch_count; ++b) {
                    int offset = 0;
                    for (int orig_idx : packed_batches[b].orig_indices) {
                        h_d_orig_idx.push_back(orig_idx);
                        h_d_batch_id.push_back(b);
                        h_d_local_offset.push_back(offset);
                        h_d_nrows.push_back(h_nrows[orig_idx]);
                        offset += h_nrows[orig_idx];
                    }
                }
                
                // Assign to class member
                num_dense_subsystems = h_d_orig_idx.size();
                
                // Allocate device mapping metadata (assigned to class members)
                CHECK_CUDA(cudaMalloc(&d_d_orig_idx, num_dense_subsystems * sizeof(int)));
                CHECK_CUDA(cudaMalloc(&d_d_batch_id, num_dense_subsystems * sizeof(int)));
                CHECK_CUDA(cudaMalloc(&d_d_local_offset, num_dense_subsystems * sizeof(int)));
                CHECK_CUDA(cudaMalloc(&d_d_nrows, num_dense_subsystems * sizeof(int)));

                CHECK_CUDA(cudaMemcpy(d_d_orig_idx, h_d_orig_idx.data(), num_dense_subsystems * sizeof(int), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_d_batch_id, h_d_batch_id.data(), num_dense_subsystems * sizeof(int), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_d_local_offset, h_d_local_offset.data(), num_dense_subsystems * sizeof(int), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_d_nrows, h_d_nrows.data(), num_dense_subsystems * sizeof(int), cudaMemcpyHostToDevice));

                // Allocate Batched Dense Matrices and Info (assigned to class members)
                size_t dense_matrix_size = dense_batch_count * max_dense_dim * max_dense_dim;
                CHECK_CUDA(cudaMalloc(&d_dense_matrices, dense_matrix_size * sizeof(double)));
                CHECK_CUDA(cudaMalloc(&d_pivots, dense_batch_count * max_dense_dim * sizeof(int)));
                CHECK_CUDA(cudaMalloc(&d_info, dense_batch_count * sizeof(int)));

                // Allocate Dense RHS/Solution Vectors for the Solve Phase (assigned to class members)
                CHECK_CUDA(cudaMalloc(&d_dense_x, dense_batch_count * max_dense_dim * sizeof(double)));
                CHECK_CUDA(cudaMalloc(&d_dense_x_ptrs, dense_batch_count * sizeof(double*)));

                // Setup host-side pointer arrays to upload to device
                std::vector<double*> h_dense_ptrs(dense_batch_count);
                std::vector<double*> h_dense_x_ptrs(dense_batch_count);
                for (int b = 0; b < dense_batch_count; ++b) {
                    h_dense_ptrs[b] = d_dense_matrices + b * (max_dense_dim * max_dense_dim);
                    h_dense_x_ptrs[b] = d_dense_x + b * max_dense_dim;
                }
                
                // Allocate device pointer arrays and copy (assigned to class members)
                CHECK_CUDA(cudaMalloc(&d_dense_ptrs, dense_batch_count * sizeof(double*)));
                CHECK_CUDA(cudaMemcpy(d_dense_ptrs, h_dense_ptrs.data(), dense_batch_count * sizeof(double*), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_dense_x_ptrs, h_dense_x_ptrs.data(), dense_batch_count * sizeof(double*), cudaMemcpyHostToDevice));

                // Lambda 1: Initialize padded space with Identity
                thrust::for_each(thrust::device,
                    thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(dense_matrix_size),
                    [=] __device__ (int tid) {
                        int rem = tid % (max_dense_dim * max_dense_dim);
                        int col = rem / max_dense_dim;
                        int row = rem % max_dense_dim;
                        d_dense_matrices[tid] = (row == col) ? 1.0 : 0.0;
                    }
                );

                // Lambda 2: Extract preassembled CSR components into dense blocks
                void** d_row_ptrs_device = d_csrRowOffsets_void;
                void** d_col_ptrs_device = d_csrColIndices_void;
                void** d_val_ptrs_device = d_csrValues_void;

                // Use local copies of class members inside the lambda to ensure clean capture
                int* local_d_orig_idx = d_d_orig_idx;
                int* local_d_batch_id = d_d_batch_id;
                int* local_d_local_offset = d_d_local_offset;
                int* local_d_nrows = d_d_nrows;
                double* local_d_dense_matrices = d_dense_matrices;
                int local_max_dense_dim = max_dense_dim;

                thrust::for_each(thrust::device,
                    thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(num_dense_subsystems),
                    [=] __device__ (int tid) {
                        int orig_idx = local_d_orig_idx[tid];
                        int batch_id = local_d_batch_id[tid];
                        int offset   = local_d_local_offset[tid];
                        int nrows    = local_d_nrows[tid];

                        int* row_ptr  = static_cast<int*>(d_row_ptrs_device[orig_idx]);
                        int* col_idx  = static_cast<int*>(d_col_ptrs_device[orig_idx]);
                        double* vals = static_cast<double*>(d_val_ptrs_device[orig_idx]);

                        for (int r = 0; r < nrows; ++r) {
                            int start = row_ptr[r];
                            int end   = row_ptr[r + 1];
                            for (int p = start; p < end; ++p) {
                                int c = col_idx[p];
                                double v = vals[p];
                                
                                // Column-major format write
                                int dense_idx = batch_id * (local_max_dense_dim * local_max_dense_dim) + (offset + c) * local_max_dense_dim + (offset + r);
                                local_d_dense_matrices[dense_idx] = v;
                            }
                        }
                    }
                );

                // Initialize class member cuBLAS handle if not already done
                if (!cublas_handle) {
                    cublasCreate(&cublas_handle);
                }

                POLYSOLVE_SCOPED_STOPWATCH("factorize DENSE", dss_factorization_time, *logger);
                int err_check = cublasDgetrfBatched(
                    cublas_handle, 
                    max_dense_dim, 
                    d_dense_ptrs,   
                    max_dense_dim, 
                    d_pivots,       
                    d_info,         
                    dense_batch_count
                );                
            }

            // ==========================================
            //            SPARSE PROCESSING (cuDSS)
            // ==========================================
            sparseBatchCount = sparse_indices.size(); // Assigned to class member
            if (sparseBatchCount > 0) {

                // Compact sparse configuration arrays
                h_sparse_nrows.resize(sparseBatchCount);
                h_sparse_ncols.resize(sparseBatchCount);
                h_sparse_nnz.resize(sparseBatchCount);
                h_sparse_vec_ncols.resize(sparseBatchCount, 1);
                h_sparse_ld.resize(sparseBatchCount);
                
                h_sparse_csrRowOffsets.resize(sparseBatchCount);
                h_sparse_csrColIndices.resize(sparseBatchCount);
                h_sparse_csrValues.resize(sparseBatchCount);
                h_sparse_x.resize(sparseBatchCount);
                h_sparse_b.resize(sparseBatchCount);

                for (int i = 0; i < sparseBatchCount; ++i) {
                    int orig_idx = sparse_indices[i];
                    
                    h_sparse_nrows[i] = h_nrows[orig_idx];
                    h_sparse_ncols[i] = h_ncols[orig_idx];
                    h_sparse_nnz[i]   = h_nnz[orig_idx];
                    h_sparse_ld[i]    = h_ld[orig_idx];

                    h_sparse_csrRowOffsets[i] = h_csrRowOffsets_void[orig_idx];
                    h_sparse_csrColIndices[i] = h_csrColIndices_void[orig_idx];
                    h_sparse_csrValues[i]     = h_csrValues_void[orig_idx];
                    h_sparse_x[i]             = h_x_void[orig_idx];
                    h_sparse_b[i]             = h_b_void[orig_idx];
                }

                void **d_sparse_csrRowOffsets, **d_sparse_csrColIndices, **d_sparse_csrValues;
                CHECK_CUDA(cudaMalloc(&d_sparse_csrRowOffsets, sparseBatchCount * sizeof(void*)));
                CHECK_CUDA(cudaMalloc(&d_sparse_csrColIndices, sparseBatchCount * sizeof(void*)));
                CHECK_CUDA(cudaMalloc(&d_sparse_csrValues, sparseBatchCount * sizeof(void*)));
                CHECK_CUDA(cudaMalloc(&d_sparse_x, sparseBatchCount * sizeof(void*)));
                CHECK_CUDA(cudaMalloc(&d_sparse_b, sparseBatchCount * sizeof(void*)));

                CHECK_CUDA(cudaMemcpy(d_sparse_csrRowOffsets, h_sparse_csrRowOffsets.data(), sparseBatchCount * sizeof(void*), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_sparse_csrColIndices, h_sparse_csrColIndices.data(), sparseBatchCount * sizeof(void*), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_sparse_csrValues, h_sparse_csrValues.data(), sparseBatchCount * sizeof(void*), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_sparse_x, h_sparse_x.data(), sparseBatchCount * sizeof(void*), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_sparse_b, h_sparse_b.data(), sparseBatchCount * sizeof(void*), cudaMemcpyHostToDevice));
                
                // cuDSS Calls
                CHECK_CUDSS(cudssMatrixCreateBatchDn(
                    &batchMatrixX, sparseBatchCount, h_sparse_nrows.data(), h_sparse_vec_ncols.data(), h_sparse_ld.data(), 
                    d_sparse_x, CUDA_R_32I, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR
                ));

                CHECK_CUDSS(cudssMatrixCreateBatchDn(
                    &batchMatrixB, sparseBatchCount, h_sparse_nrows.data(), h_sparse_vec_ncols.data(), h_sparse_ld.data(), 
                    d_sparse_b, CUDA_R_32I, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR
                ));

                {
                    POLYSOLVE_SCOPED_STOPWATCH("factorize D", dss_factorization_time, *logger);

                    CHECK_CUDSS(cudssMatrixCreateBatchCsr(
                        &batchMatrixA, sparseBatchCount, h_sparse_nrows.data(), h_sparse_ncols.data(), h_sparse_nnz.data(), 
                        d_sparse_csrRowOffsets, nullptr, d_sparse_csrColIndices, d_sparse_csrValues, 
                        CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SYMMETRIC, 
                        CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO
                    ));

                    CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_ANALYSIS, config, solverData, 
                                            batchMatrixA, nullptr, nullptr));
                    CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_FACTORIZATION, config, solverData, 
                                        batchMatrixA, nullptr, nullptr));

                    int64_t lu_nnz = 0;
                    size_t bytes_written = 0;

                    CHECK_CUDSS(cudssDataGet(cudss_handle, solverData, CUDSS_DATA_LU_NNZ, 
                                        &lu_nnz, sizeof(int64_t), &bytes_written));

                    logger->trace("Total LU nnz: {}", lu_nnz);
                }
                
                // Free compacted sparse device pointers after cuDSS matrix creation
                cudaFree(d_sparse_csrRowOffsets);
                cudaFree(d_sparse_csrColIndices);
                cudaFree(d_sparse_csrValues);
                CHECK_CUDA(cudaDeviceSynchronize());
            }

        }

    }

    void GPUHybrid::pcg_solve(double* rhs, double* result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond)
    {
        double pre_loop_time;
        double bi_prod, eps, gamma, old_gamma;

        double* r;
        double* p;
        double* z;
        double* z2;
        double* buffer;

        CHECK_CUDA(cudaMalloc(&r, sparse_A.rows() * sizeof(double)));
        CHECK_CUDA(cudaMemset(r, 0, sparse_A.rows() * sizeof(double)));

        CHECK_CUDA(cudaMalloc(&p, sparse_A.rows() * sizeof(double)));
        CHECK_CUDA(cudaMemset(p, 0, sparse_A.rows() * sizeof(double)));

        CHECK_CUDA(cudaMalloc(&z, sparse_A.rows() * sizeof(double)));
        CHECK_CUDA(cudaMemset(z, 0, sparse_A.rows() * sizeof(double)));

        CHECK_CUDA(cudaMalloc(&z2, sparse_A.rows() * sizeof(double)));
        CHECK_CUDA(cudaMemset(z2, 0, sparse_A.rows() * sizeof(double)));

        CHECK_CUDA(cudaMalloc(&buffer, sparse_A.rows() * sizeof(double)));
        CHECK_CUDA(cudaMemset(buffer, 0, sparse_A.rows() * sizeof(double)));

        {
            POLYSOLVE_SCOPED_STOPWATCH("pre loop time: ", pre_loop_time, *logger);
        
            bi_prod = dot(rhs, rhs);
            logger->trace("GPUHybrid solver bi prod: {}", bi_prod);

            if (bi_prod > 0.0)
            {
                eps = conv_tol_ * conv_tol_;
            }
            else 
            {
                CHECK_CUDA(cudaMemset(result, 0, sparse_A.rows() * sizeof(double)));
                num_iterations = 0;
                final_res_norm = 0;
                CHECK_CUDA(cudaFree(r));
                CHECK_CUDA(cudaFree(p));
                CHECK_CUDA(cudaFree(z));
                CHECK_CUDA(cudaFree(z2));
                CHECK_CUDA(cudaFree(buffer));
                return;
            }

            Eigen::VectorXd A_times_result;
            matmul(result, buffer);

            
            vector_copy(rhs, r);
            vector_add(-1.0, buffer, r);

            custom_mixed_precond_iter(precond, r, z, buffer, z2);

            vector_copy(z, p);

            gamma = dot(r, z);
            old_gamma = gamma;
        }

        for (int k = 0; k < max_iter_; ++k)
        {
            POLYSOLVE_SCOPED_STOPWATCH("main loop time: ", loop_time, *logger);
            num_iterations = k + 1;

            matmul(p, buffer);
            double sdotp = dot(p, buffer);

            if (sdotp == 0.0)
            {
                logger->debug("GPUHybrid solver error: zero sdotp value");
                break;
            }

            double alpha = gamma / sdotp;

            if (alpha <= 0.0)
            {
                logger->debug("GPUHybrid solver error: negative or zero alpha value. gamma: {}, sdotp: {}", gamma, sdotp);
                break;
            } 
            else if (alpha < __DBL_MIN__)
            {
                logger->debug("GPUHybrid solver error: subnormal alpha value");
                break;
            }

            vector_add(alpha, p, result);
            vector_add(-1.0 * alpha, buffer, r);
            
            double drob2 = alpha * alpha * dot(p, p);
            if (!use_absolute_tol) 
            {
                drob2 /= bi_prod;
            }

            if (drob2 < conv_tol_ * conv_tol_)
            {
                logger->debug("GPUHybrid solver converged: change in residual too small");
                //break;
            }

            double i_prod = dot(r, r);
            logger->trace("GPUHybrid solver i prod: {}", i_prod);
            if (!use_absolute_tol) 
            {
                i_prod /= bi_prod;
            }

            if (i_prod < eps)
            {
                logger->debug("GPUHybrid solver converged: residual too small");
                break;
            }

            CHECK_CUDA(cudaMemset(z, 0, sparse_A.rows() * sizeof(double)));
            custom_mixed_precond_iter(precond, r, z, buffer, z2);

            gamma = dot(r, z);
            
            double beta = gamma / old_gamma;
            old_gamma = gamma;

            vector_scale(beta, p);
            vector_add(1.0, z, p);
        }
        CHECK_CUDA(cudaFree(r));
        CHECK_CUDA(cudaFree(p));
        CHECK_CUDA(cudaFree(z));
        CHECK_CUDA(cudaFree(z2));
        CHECK_CUDA(cudaFree(buffer));
    }

    GPUHybrid::~GPUHybrid()
    {
        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }
        free_device_memory();
        if (cudss_handle) {
            cudssDestroy(cudss_handle);
            cudss_handle = nullptr;
        }
        if (cublas_handle) {
            cublasDestroy(cublas_handle);
            cublas_handle = nullptr;
        }
    }

    void GPUHybrid::free_device_memory() 
    {
        // Destroy cuDSS Opaque Structures
        if (batchMatrixA) { CHECK_CUDSS(cudssMatrixDestroy(batchMatrixA)); batchMatrixA = nullptr; }
        if (batchMatrixX) { CHECK_CUDSS(cudssMatrixDestroy(batchMatrixX)); batchMatrixX = nullptr; }
        if (batchMatrixB) { CHECK_CUDSS(cudssMatrixDestroy(batchMatrixB)); batchMatrixB = nullptr; }
        if (solverData)   { CHECK_CUDSS(cudssDataDestroy(cudss_handle, solverData)); solverData = nullptr; }
        if (config)       { CHECK_CUDSS(cudssConfigDestroy(config)); config = nullptr; }

        if (d_all_rowOffsets)     { CHECK_CUDA(cudaFree(d_all_rowOffsets)); d_all_rowOffsets = nullptr; }
        if (d_all_colIndices)     { CHECK_CUDA(cudaFree(d_all_colIndices)); d_all_colIndices = nullptr; }
        if (d_all_values)         { CHECK_CUDA(cudaFree(d_all_values)); d_all_values = nullptr; }
        
        if (d_csrRowOffsets_void) { CHECK_CUDA(cudaFree(d_csrRowOffsets_void)); d_csrRowOffsets_void = nullptr; }
        if (d_csrColIndices_void) { CHECK_CUDA(cudaFree(d_csrColIndices_void)); d_csrColIndices_void = nullptr; }
        if (d_csrValues_void)     { CHECK_CUDA(cudaFree(d_csrValues_void)); d_csrValues_void = nullptr; }
        if (d_x_void)             { CHECK_CUDA(cudaFree(d_x_void)); d_x_void = nullptr; }
        if (d_b_void)             { CHECK_CUDA(cudaFree(d_b_void)); d_b_void = nullptr; }

        // Free GPU Data Arrays
        if (d_x)             { cudaFree(d_x); d_x = nullptr; }
        if (d_b)             { cudaFree(d_b); d_b = nullptr; }

        if (d_max_estimated_eigenvalues) { cudaFree(d_max_estimated_eigenvalues); d_max_estimated_eigenvalues = nullptr; }
        if (d_matrix_dof_starts) { cudaFree(d_matrix_dof_starts); d_matrix_dof_starts = nullptr; }
    
        if (d_x_curr) { cudaFree(d_x_curr); d_x_curr = nullptr; }
        if (d_x_prev) { cudaFree(d_x_prev); d_x_prev = nullptr; }
        if (d_x_new) { cudaFree(d_x_new); d_x_new = nullptr; }
        if (d_d) { cudaFree(d_d); d_d = nullptr; }
        if (d_rho_sq) { cudaFree(d_rho_sq); d_rho_sq = nullptr; }
        if (d_omega) { cudaFree(d_omega); d_omega = nullptr; }

        if (d_d_orig_idx)     { cudaFree(d_d_orig_idx);     d_d_orig_idx = nullptr; }
        if (d_d_batch_id)     { cudaFree(d_d_batch_id);     d_d_batch_id = nullptr; }
        if (d_d_local_offset) { cudaFree(d_d_local_offset); d_d_local_offset = nullptr; }
        if (d_d_nrows)        { cudaFree(d_d_nrows);        d_d_nrows = nullptr; }

        // 2. Free Dense Factorization Data
        if (d_dense_matrices) { cudaFree(d_dense_matrices); d_dense_matrices = nullptr; }
        if (d_dense_ptrs)     { cudaFree(d_dense_ptrs);     d_dense_ptrs = nullptr; }
        if (d_pivots)         { cudaFree(d_pivots);         d_pivots = nullptr; }
        if (d_info)           { cudaFree(d_info);           d_info = nullptr; }

        // 3. Free Dense RHS/Solve Vectors
        if (d_dense_x)        { cudaFree(d_dense_x);        d_dense_x = nullptr; }
        if (d_dense_x_ptrs)   { cudaFree(d_dense_x_ptrs);   d_dense_x_ptrs = nullptr; }

        if (d_sparse_x)        { cudaFree(d_sparse_x);        d_sparse_x = nullptr; }
        if (d_sparse_b)        { cudaFree(d_sparse_b);        d_sparse_b = nullptr; }
    }
}