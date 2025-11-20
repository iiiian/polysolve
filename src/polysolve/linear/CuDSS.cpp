#ifdef POLYSOLVE_WITH_CUSOLVER

////////////////////////////////////////////////////////////////////////////////
#include "CuDSS.hpp"

#include <cudss.h>
#include <cuda_runtime_api.h>

#include <Eigen/Sparse>

////////////////////////////////////////////////////////////////////////////////

namespace polysolve::linear
{

#ifdef POLYSOLVE_LARGE_INDEX
    using EigenIndexType = std::ptrptrdiff_t;
    constexpr cudaDataType_t CudaIndexType = CUDA_R_64I;
#else
    using EigenIndexType = int;
    constexpr cudaDataType_t CudaIndexType = CUDA_R_32I;
#endif

    void CuDSS::check_cuda(cudaError_t code, const char *msg)
    {
        if (code != cudaSuccess)
        {
            throw std::runtime_error(std::string("[CuDSS] CUDA error in ") + msg + ": " + cudaGetErrorString(code));
        }
    }

    void CuDSS::check_cudss(cudssStatus_t status, const char *msg)
    {
        if (status != CUDSS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::string("[CuDSS] cuDSS error in ") + msg);
        }
    }

    CuDSS::CuDSS()
    {
        check_cudss(cudssCreate(&handle_), "cudssCreate");

        // Create stream and bind to handle
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        check_cudss(cudssSetStream(handle_, stream_), "cudssSetStream");

        check_cudss(cudssConfigCreate(&config_), "cudssConfigCreate");
        check_cudss(cudssDataCreate(handle_, &data_), "cudssDataCreate");
    }

    CuDSS::~CuDSS()
    {
        free_matrix();

        if (b_dn_ != nullptr)
        {
            cudssMatrixDestroy(b_dn_);
            b_dn_ = nullptr;
        }
        if (x_dn_ != nullptr)
        {
            cudssMatrixDestroy(x_dn_);
            x_dn_ = nullptr;
        }

        if (d_b_ != nullptr)
        {
            check_cuda(cudaFree(d_b_), "cudaFree d_b_");
            d_b_ = nullptr;
        }
        if (d_x_ != nullptr)
        {
            check_cuda(cudaFree(d_x_), "cudaFree d_x_");
            d_x_ = nullptr;
        }

        if (data_ != nullptr)
        {
            cudssDataDestroy(handle_, data_);
            data_ = nullptr;
        }
        if (config_ != nullptr)
        {
            cudssConfigDestroy(config_);
            config_ = nullptr;
        }
        if (handle_ != nullptr)
        {
            cudssDestroy(handle_);
            handle_ = nullptr;
        }
        if (stream_ != nullptr)
        {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////

    void CuDSS::free_matrix()
    {
        if (A_csr_ != nullptr)
        {
            cudssMatrixDestroy(A_csr_);
            A_csr_ = nullptr;
        }

        if (d_rowOffsets_ != nullptr)
        {
            check_cuda(cudaFree(d_rowOffsets_), "cudaFree d_rowOffsets_");
            d_rowOffsets_ = nullptr;
        }
        if (d_colIndices_ != nullptr)
        {
            check_cuda(cudaFree(d_colIndices_), "cudaFree d_colIndices_");
            d_colIndices_ = nullptr;
        }
        if (d_values_ != nullptr)
        {
            check_cuda(cudaFree(d_values_), "cudaFree d_values_");
            d_values_ = nullptr;
        }

        nrows_ = 0;
        ncols_ = 0;
        nnz_ = 0;
        analyzed_ = false;
        factorized_ = false;
    }

    void CuDSS::allocate_matrix(const StiffnessMatrix &A)
    {
        // Convert to row-major CSR
        Eigen::SparseMatrix<double, Eigen::RowMajor, EigenIndexType> A_row = A;
        A_row.makeCompressed();

        const int64_t nrows = static_cast<int64_t>(A_row.rows());
        const int64_t ncols = static_cast<int64_t>(A_row.cols());
        const int64_t nnz = static_cast<int64_t>(A_row.nonZeros());

        // If dimensions changed, free and reallocate
        if (A_csr_ != nullptr && (nrows != nrows_ || ncols != ncols_ || nnz != nnz_))
        {
            free_matrix();
        }

        nrows_ = nrows;
        ncols_ = ncols;
        nnz_ = nnz;

        if (d_rowOffsets_ == nullptr)
        {
            check_cuda(cudaMalloc(&d_rowOffsets_, sizeof(EigenIndexType) * (nrows_ + 1)), "cudaMalloc d_rowOffsets_");
        }
        if (d_colIndices_ == nullptr)
        {
            check_cuda(cudaMalloc(&d_colIndices_, sizeof(EigenIndexType) * nnz_), "cudaMalloc d_colIndices_");
        }
        if (d_values_ == nullptr)
        {
            check_cuda(cudaMalloc(&d_values_, sizeof(double) * nnz_), "cudaMalloc d_values_");
        }

        // Upload structure and values
        check_cuda(cudaMemcpy(d_rowOffsets_, A_row.outerIndexPtr(), sizeof(EigenIndexType) * (nrows_ + 1), cudaMemcpyHostToDevice), "cudaMemcpy rowOffsets");
        check_cuda(cudaMemcpy(d_colIndices_, A_row.innerIndexPtr(), sizeof(EigenIndexType) * nnz_, cudaMemcpyHostToDevice), "cudaMemcpy colIndices");
        check_cuda(cudaMemcpy(d_values_, A_row.valuePtr(), sizeof(double) * nnz_, cudaMemcpyHostToDevice), "cudaMemcpy values");

        // Create matrix wrapper if needed
        if (A_csr_ == nullptr)
        {
            cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
            cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
            cudssIndexBase_t base = CUDSS_BASE_ZERO;

            check_cudss(
                cudssMatrixCreateCsr(
                    &A_csr_, nrows_, ncols_, nnz_,
                    d_rowOffsets_, nullptr,
                    d_colIndices_, d_values_,
                    CudaIndexType, CUDA_R_64F,
                    mtype, mview, base),
                "cudssMatrixCreateCsr");
        }
    }

    void CuDSS::upload_values(const StiffnessMatrix &A)
    {
        Eigen::SparseMatrix<double, Eigen::RowMajor, EigenIndexType> A_row = A;
        A_row.makeCompressed();

        const int64_t nnz = static_cast<int64_t>(A_row.nonZeros());
        if (nnz != nnz_ || A_row.rows() != nrows_ || A_row.cols() != ncols_)
        {
            // Pattern changed: reallocate
            allocate_matrix(A);
            return;
        }

        check_cuda(cudaMemcpy(d_values_, A_row.valuePtr(), sizeof(double) * nnz_, cudaMemcpyHostToDevice), "cudaMemcpy values (update)");
    }

    void CuDSS::ensure_dense_wrappers(int64_t nrows, int nrhs)
    {
        // Reallocate device buffers for RHS and solution for given dimensions
        if (d_b_ != nullptr)
        {
            check_cuda(cudaFree(d_b_), "cudaFree d_b_");
            d_b_ = nullptr;
        }
        if (d_x_ != nullptr)
        {
            check_cuda(cudaFree(d_x_), "cudaFree d_x_");
            d_x_ = nullptr;
        }

        check_cuda(cudaMalloc(&d_b_, sizeof(double) * nrows * nrhs), "cudaMalloc d_b_");
        check_cuda(cudaMalloc(&d_x_, sizeof(double) * nrows * nrhs), "cudaMalloc d_x_");

        if (b_dn_ != nullptr)
        {
            cudssMatrixDestroy(b_dn_);
            b_dn_ = nullptr;
        }
        if (x_dn_ != nullptr)
        {
            cudssMatrixDestroy(x_dn_);
            x_dn_ = nullptr;
        }

        const int64_t nrows_mat = nrows;
        const int64_t ncols_mat = nrhs;
        const int64_t ld = nrows;

        check_cudss(
            cudssMatrixCreateDn(
                &b_dn_, nrows_mat, ncols_mat, ld,
                d_b_, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
            "cudssMatrixCreateDn b");

        check_cudss(
            cudssMatrixCreateDn(
                &x_dn_, nrows_mat, ncols_mat, ld,
                d_x_, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
            "cudssMatrixCreateDn x");
    }

    ////////////////////////////////////////////////////////////////////////////////

    void CuDSS::analyze_pattern(const StiffnessMatrix &A, const int precond_num)
    {
        (void)precond_num;

        allocate_matrix(A);

        // For analysis, we do not need actual RHS/solution values, but cuDSS expects
        // valid matrix objects. Create 1-column dense wrappers with dummy data.
        ensure_dense_wrappers(nrows_, 1);

        check_cudss(
            cudssExecute(
                handle_,
                CUDSS_PHASE_ANALYSIS,
                config_, data_,
                A_csr_, x_dn_, b_dn_),
            "cudssExecute ANALYSIS");

        analyzed_ = true;
        factorized_ = false;
    }

    void CuDSS::factorize(const StiffnessMatrix &A)
    {
        if (!analyzed_)
        {
            analyze_pattern(A, static_cast<int>(A.rows()));
            return;
        }

        upload_values(A);

        check_cudss(
            cudssExecute(
                handle_,
                CUDSS_PHASE_FACTORIZATION,
                config_, data_,
                A_csr_, x_dn_, b_dn_),
            "cudssExecute FACTORIZATION");

        factorized_ = true;
    }

    void CuDSS::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        if (!factorized_)
        {
            throw std::runtime_error("[CuDSS] factorize must be called before solve.");
        }
        if (b.size() != nrows_)
        {
            throw std::runtime_error("[CuDSS] RHS size does not match matrix dimension.");
        }

        if (x.size() != b.size())
        {
            x.resize(b.size());
        }

        const int nrhs = 1;
        ensure_dense_wrappers(nrows_, nrhs);

        // Copy RHS to device
        check_cuda(cudaMemcpy(d_b_, b.data(), sizeof(double) * b.size(), cudaMemcpyHostToDevice), "cudaMemcpy b");

        // Solve
        check_cudss(
            cudssExecute(
                handle_,
                CUDSS_PHASE_SOLVE,
                config_, data_,
                A_csr_, x_dn_, b_dn_),
            "cudssExecute SOLVE");

        // Copy solution back
        check_cuda(cudaMemcpy(x.data(), d_x_, sizeof(double) * x.size(), cudaMemcpyDeviceToHost), "cudaMemcpy x");
    }

    ////////////////////////////////////////////////////////////////////////////////

} // namespace polysolve::linear

#endif
