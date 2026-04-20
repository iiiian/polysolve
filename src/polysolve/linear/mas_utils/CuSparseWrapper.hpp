#pragma once

#include <cusparse.h>
#include <polysolve/linear/mas_utils/BSRMatrix.hpp>

namespace polysolve::linear::mas
{
    /// @brief Minimal RAII wrapper for CuSparse dense vector.
    class CuSparseVec
    {
    public:
        cusparseConstDnVecDescr_t raw;

        CuSparseVec(ctd::span<double> vec)
        {
            cusparseCreateConstDnVec(&raw, vec.size(), vec.data(), CUDA_R_64F);
        }

        ~CuSparseVec()
        {
            cusparseDestroyDnVec(raw);
        }

        CuSparseVec(const BSRView &) = delete;
        CuSparseVec(CuSparseVec &&other) { swap(other); }
        CuSparseVec &operator=(const BSRView &) = delete;
        CuSparseVec &operator=(CuSparseVec &&other) { swap(other); }

    private:
        void swap(CuSparseVec &other)
        {
            std::swap(raw, other.raw);
        }
    };

    /// @brief Minimal RAII wrapper for CuSparse BSR matrix.
    class CuSparseBSR
    {
    public:
        cusparseConstSpMatDescr_t raw;

        CuSparseBSR(BSRView mat)
        {
            cusparseCreateConstBsr(&raw,
                                   mat.dim,
                                   mat.dim,
                                   mat.non_zeros,
                                   mat.block_dim,
                                   mat.block_dim,
                                   mat.rows.data(),
                                   mat.cols.data(),
                                   mat.vals.data(),
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_64F,
                                   CUSPARSE_ORDER_ROW);
        }

        ~CuSparseBSR()
        {
            cusparseDestroySpMat(raw);
        }

        CuSparseBSR(const BSRView &) = delete;
        CuSparseBSR(CuSparseBSR &&other) { swap(other); }
        CuSparseBSR &operator=(const BSRView &) = delete;
        CuSparseBSR &operator=(CuSparseBSR &&other) { swap(other); }

    private:
        void swap(CuSparseBSR &other)
        {
            std::swap(raw, other.raw);
        }
    };

} // namespace polysolve::linear::mas
