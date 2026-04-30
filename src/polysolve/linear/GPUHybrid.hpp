#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"

#include <vector>
#include <deque>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>

#include <cudss.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>


extern "C" {
    HYPRE_Int hypre_ParVectorAxpy(HYPRE_Complex alpha, HYPRE_ParVector x, HYPRE_ParVector y);
}

namespace polysolve::linear
{
    class GPUHybrid : public Solver
    {

    public:
        GPUHybrid();
        ~GPUHybrid();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(GPUHybrid)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void set_parameters(const json &params) override;

        // Retrieve solve information
        virtual void get_info(json &params) const override;

        void check_settings() const;

        // Analyze sparsity pattern
        virtual void analyze_pattern(const StiffnessMatrix &A, const int precond_num) override;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { 
            if (do_mixed_precond)
            {
                if (select_bad_dofs_from_l1_row_norm)
                {
                    return "GPUHybrid";
                }
                return "GPUAMGF";
            }
            return "GPUAMG"; 
        }

        //virtual void set_problematic_dofs(const std::set<int> &bad_dofs) override {TODO}

    protected:
        // AMG settings
        double theta = 0.5;

        // Hybrid preconditioner settings
        bool select_bad_dofs_from_l1_row_norm = true;
        bool decompose_subdomains = false;
        int min_subdomain_size = 1;
        int max_subdomain_size = 1e9;
        double gmm_bic_threshold = 0.1;
        int max_dense_size = 1000;
        bool use_float_on_subdomains = false;
        bool expand_subdomains = true;

        // General solver settings
        int dimension_ = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)
        int max_iter_ = 10000;
        double rel_conv_tol_ = 1e-10;
        double abs_conv_tol_ = 0.0;
        bool do_mixed_precond = false;

    private:
        bool has_matrix_ = false;

        // problem-specific data
        Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_A;

        // Hypre variables
        HYPRE_IJMatrix A;
        HYPRE_ParCSRMatrix parcsr_A;
        HYPRE_IJVector ij_x;
        HYPRE_IJVector ij_b;

        // hybrid preconditioner data
        std::vector<thrust::device_vector<int>> bad_indices_arrays;
        thrust::device_vector<int> all_bad_dof_map;

        // cudss data
        // cuDSS Handles and Descriptors
        cudssHandle_t cudss_handle = nullptr;
        cudssConfig_t config = nullptr;
        cudssData_t solverData = nullptr;
        
        cudssMatrix_t batchMatrixA = nullptr;
        cudssMatrix_t batchMatrixX = nullptr;
        cudssMatrix_t batchMatrixB = nullptr;

        cublasHandle_t cublas_handle = nullptr;

        int dense_batch_count = 0;
        int num_dense_subsystems = 0;
        int max_dense_dim = 0;
        int sparseBatchCount = 0;

        // --- Dense Subsystem Mapping (Device Pointers) ---
        int* d_d_orig_idx = nullptr;
        int* d_d_batch_id = nullptr;
        int* d_d_local_offset = nullptr;
        int* d_d_nrows = nullptr;

        // --- Dense Factorization & Solve Data (Device Pointers) ---
        double* d_dense_matrices = nullptr; // Padded matrices (overwritten with LU)
        double** d_dense_ptrs = nullptr;    // Array of pointers to each matrix
        int* d_pivots = nullptr;            // Pivot arrays from GETRF
        int* d_info = nullptr;              // Error info from cuBLAS/cuSolver
        
        double* d_dense_x = nullptr;        // Padded RHS/Solution vectors
        double** d_dense_x_ptrs = nullptr;  // Array of pointers to each RHS vector

        void free_device_memory();

        // Dimensions (kept as class members to ensure pointers survive across phases)
        int m_nrows = 0;
        int m_ncols = 0;
        int m_nnz = 0;
        int m_batchCount = 1;

        std::vector<int> h_nrows;
        std::vector<int> h_ncols;
        std::vector<int> h_nnz;
        std::vector<int> h_vec_ncols;
        std::vector<int> h_ld;

        std::vector<void*> h_csrRowOffsets_void;
        std::vector<void*> h_csrColIndices_void;
        std::vector<void*> h_csrValues_void;
        std::vector<void*> h_x_void;
        std::vector<void*> h_b_void;

        // Device memory pointers (Arrays)
        int* d_csrRowOffsets = nullptr;
        int* d_csrColIndices = nullptr;
        double* d_csrValues = nullptr;
        double* d_x = nullptr;
        double* d_b = nullptr;

        int* d_all_rowOffsets = nullptr;
        int* d_all_colIndices = nullptr;
        double* d_all_values = nullptr;

        // Device memory pointers (Arrays of pointers for cuDSS Batch API)
        void **d_csrRowOffsets_void = nullptr;
        void **d_csrColIndices_void = nullptr;
        void **d_csrValues_void = nullptr;
        void **d_x_void = nullptr;
        void **d_b_void = nullptr;

        void **d_sparse_x = nullptr;
        void **d_sparse_b = nullptr;

        int* d_matrix_dof_starts = nullptr; // For binary search in kernels

        double* d_x_curr;
        double* d_x_prev;
        double* d_x_new;
        double* d_d;
        double* d_rho_sq;
        double* d_omega;

        std::vector<int> row_starts;
        std::vector<int> nnz_starts;
        std::vector<int> dof_starts;

        double* d_max_estimated_eigenvalues = nullptr;

        std::vector<int> h_sparse_nrows;
        std::vector<int> h_sparse_ncols;
        std::vector<int> h_sparse_nnz;
        std::vector<int> h_sparse_vec_ncols;
        std::vector<int> h_sparse_ld;
        
        std::vector<void*> h_sparse_csrRowOffsets;
        std::vector<void*> h_sparse_csrColIndices;
        std::vector<void*> h_sparse_csrValues;
        std::vector<void*> h_sparse_x;
        std::vector<void*> h_sparse_b;

    public:
        void copy_matrix_to_hypre();

        // solve helpers
        void init_hypre_vectors(const int size);
        void set_hypre_vec(HYPRE_IJVector &ij_x, HYPRE_ParVector &par_x, double* x);

        // linear algebra helpers
        void matmul(double* x, double* result);
        double dot(double* x, double* y);
        void vector_copy(double* x, double* y);
        void vector_add(double alpha, double* x, double* y);
        void vector_scale(double alpha, double* x);

        // preconditioning functions
        void custom_mixed_precond_iter(const HYPRE_Solver &precond, double* r, double* z, double* buffer, double* z2);
        void amg_precond_iter(const HYPRE_Solver &precond, double* b, double* x);
        void dss_precond_iter(double* z, double* r, double* next_z);

        // hybrid preconditioner preparation functions
        void prepare_dss();
        void decompose_subdomains_to_disjoint_subsets(std::vector<std::set<int>> &overlap_extensions);
        void select_bad_indices();
        void factorize_submatrix();
        void assemble_D(int bad_i, int i, Eigen::SparseMatrix<double, Eigen::RowMajor>& D);
        void allocate_subdomains();

        // Krylov solve methods
        void pcg_solve(double* rhs, double* result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond);

    };

} // namespace polysolve::linear