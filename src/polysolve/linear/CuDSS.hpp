#pragma once

#ifdef POLYSOLVE_WITH_CUSOLVER

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"

#include <cuda_runtime.h>
#include <cudss.h>

////////////////////////////////////////////////////////////////////////////////
//
// cuDSS sparse direct solver wrapper.
//

namespace polysolve::linear
{

    class CuDSS : public Solver
    {
    public:
        CuDSS();
        ~CuDSS();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(CuDSS)

    public:
        // Analyze sparsity pattern and perform symbolic factorization
        void analyze_pattern(const StiffnessMatrix &A, const int precond_num) override;

        // Factorize system matrix (numeric factorization)
        void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        std::string name() const override { return "cuDSS"; }

    private:
        void to_device_CSR(const StiffnessMatrix &A);
        void free_matrix();
        void ensure_dense_wrappers(int64_t nrows, int nrhs);

        void upload_values(const StiffnessMatrix &A);

        static void check_cuda(cudaError_t code, const char *msg);
        static void check_cudss(cudssStatus_t status, const char *msg);

    private:
        // cuDSS objects
        cudssHandle_t handle_ = nullptr;
        cudssConfig_t config_ = nullptr;
        cudssData_t data_ = nullptr;

        // CUDA stream used by cuDSS
        cudaStream_t stream_ = nullptr;

        // Matrix wrappers
        cudssMatrix_t A_csr_ = nullptr;
        cudssMatrix_t x_dn_ = nullptr;
        cudssMatrix_t b_dn_ = nullptr;

        // Device storage for CSR matrix
        void *d_rowOffsets_ = nullptr;
        void *d_colIndices_ = nullptr;
        double *d_values_ = nullptr;

        // Device storage for RHS and solution
        double *d_b_ = nullptr;
        double *d_x_ = nullptr;

        // Matrix sizes
        int64_t nrows_ = 0;
        int64_t ncols_ = 0;
        int64_t nnz_ = 0;

        // Flags
        bool analyzed_ = false;
        bool factorized_ = false;
    };

} // namespace polysolve::linear

#endif
