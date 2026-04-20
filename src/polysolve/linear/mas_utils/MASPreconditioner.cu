#include <polysolve/linear/mas_utils/MASPreconditioner.hpp>

#if 0

#include <polysolve/linear/mas_utils/CudaUtils.cuh>
#include <polysolve/linear/mas_utils/MetisPartition.hpp>
#include <polysolve/linear/mas_utils/BSRMatrix.hpp>

#include <cub/cub.cuh>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/atomic>
#include <cuda/warp>
#include <cuda/std/bit>
#include <ctd/array>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace polysolve::linear::mas
{
    namespace
    {
        constexpr int MAX_COARSE_LEVEL = 4;

        struct CoarseSpace
        {
            cu::device_buffer<int> map;
            ctd::array<int, MAX_COARSE_LEVEL> cco_nums;
        };

        struct CoarseMatrices
        {
            int mat_dim;
            int mat_storage_size;
            ctd::array<ctd::span<double>, MAX_COARSE_LEVEL> matrix_per_level;
        };

        /// @brief Get coarse space CCO id.
        /// @param map Coarse space map.
        /// @param vid Vertex id.
        /// @param level Coarse space level. Start at lv 0.
        /// @return CCO id.
        __both__ int get_coarse_space_id(ctd::span<const int> map, int vid, int level)
        {
            assert(level >= 0);
            return map[vid * MAX_COARSE_LEVEL + level];
        }

        /// @brief Build local CCO mapping from input space -> coarse space lv 0 and collapse topology.
        /// @param row_ptr CSR graph topology (does not include self).
        /// @param cols CSR graph topology (does not include self).
        /// @param cco_num_per_bank Independent CCO number per bank.
        /// @param local_cco_ids Bank local CCO id at coarse space lv 0.
        __global__ void build_local_cco_lv0(ctd::span<const int> row_ptr,
                                            ctd::span<int> cols,
                                            ctd::span<int> cco_num_per_bank,
                                            ctd::span<int> local_cco_ids)
        {
            // Bank local neighbor masks.
            __shared__ uint32_t neighbors[128];

            int btid = threadIdx.x;                   // block local thread id
            int tid = blockDim.x * blockIdx.x + btid; // global thread id
            int wid = threadIdx.x / 32;               // block local warp id
            int lid = threadIdx.x % 32;               // lane id
            int bid = tid / 32;                       // bank id

            int node_num = row_ptr.size() - 1;
            if (tid >= node_num)
            {
                return;
            }

            // Build neighbor mask (including self) and collapse topology.
            uint32_t neighbor = (1u << lid);
            int out_of_bank_neighbor_count = 0;
            for (int n = row_ptr[tid]; n < row_ptr[tid + 1]; ++n)
            {
                int neighbor_vid = cols[n];
                int neighbor_bid = neighbor_vid / 32;
                if (bid == neighbor_bid)
                {
                    neighbor |= (1u << (neighbor_vid % 32));
                }
                else
                {
                    // Compact topology to include out of bank neighbors exclusively. So we dont
                    // need additional filtering when building future coarse spaces.
                    cols[out_of_bank_neighbor_count] = neighbor_vid;
                    ++out_of_bank_neighbor_count;
                }
            }
            if (out_of_bank_neighbor_count != (row_ptr[tid + 1] - row_ptr[tid]))
            {
                // -1 denotes neighbor list ending.
                cols[out_of_bank_neighbor_count] = -1;
            }
            neighbors[btid] = neighbor;
            __syncwarp();

            // Build connectivity mask (including self).
            uint32_t connection = neighbor;
            uint32_t visited = (1u << lid);
            uint32_t to_visit = connection ^ visited;
            while (to_visit)
            {
                int visiting = ctd::countr_one(to_visit);
                connection |= neighbors[visiting + wid * 32];
                visited |= (1u << visiting);
                to_visit = connection ^ visited;
            }

            // Find bank (warp) local connected component (CCO).
            uint32_t cco_lead_lid = ctd::countr_one(connection);
            bool is_lead = (cco_lead_lid == lid);
            uint32_t lead_lanes = __ballot_sync(0xFFFFFFFFu, is_lead);
            uint32_t before_cco_lead_mask = static_cast<uint32_t>((1ull << cco_lead_lid) - 1ull);

            // Write cco info back.
            local_cco_ids[tid] = ctd::popcount(lead_lanes | before_cco_lead_mask);
            if (lid == 0)
            {
                int bank_cco_num = ctd::popcount(lead_lanes);
                cco_num_per_bank[bid] = bank_cco_num;
            }
        }

        /// @brief Build bank neighbor mask and collapse topology.
        /// @param level Target coarse space level.
        /// @param coarse_space_map Coarse space map.
        /// @param row_ptr CSR graph topology (does not include self).
        /// @param cols CSR graph topology (does not include self).
        /// @param neighbors Bank local neighbor mask.
        __global__ void build_neighbor_masks_lvx(int level,
                                                 ctd::span<const int> coarse_space_map,
                                                 ctd::span<const int> row_ptr,
                                                 ctd::span<int> cols,
                                                 ctd::span<uint32_t> neighbors)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread id
            int node_num = row_ptr.size() - 1;
            if (tid >= node_num)
            {
                return;
            }

            // Build neighbor mask (including self) and collapse topology.
            int cco_id = get_coarse_space_id(coarse_space_map, tid, level - 1);
            uint32_t neighbor = (1u << (cco_id % 32));
            int out_of_bank_neighbor_count = 0;
            for (int n = row_ptr[tid]; n < row_ptr[tid + 1]; ++n)
            {
                int neighbor_vid = cols[n];
                if (neighbor_vid == -1)
                {
                    break;
                }

                int neighbor_cco_id = get_coarse_space_id(coarse_space_map, tid, level - 1);
                if (cco_id / 32 == neighbor_cco_id / 32)
                {
                    neighbor |= (1u << (neighbor_cco_id % 32));
                }
                else
                {
                    cols[out_of_bank_neighbor_count] = neighbor_vid;
                    ++out_of_bank_neighbor_count;
                }
            }
            if (out_of_bank_neighbor_count != (row_ptr[tid + 1] - row_ptr[tid]))
            {
                cols[out_of_bank_neighbor_count] = -1;
            }

            // TODO: maybe optimize?
            // 1. write cco id to shared mem [128]
            // 2. block sort
            // 3. count block local cco id
            // 4. atomic or to block local cco neighbor slot.
            cu::atomic_ref<uint32_t> neighbor_out{neighbors[cco_id]};
            neighbor_out.fetch_or(neighbor);
        }

        /// @brief Build local CCO mapping from coarse space level x-1 -> coarse space level x.
        /// @param level Target coarse space lv x.
        /// @param cco_num Total CCO number at coarse space lv x-1.
        /// @param cco_num_per_bank Independent CCO number per bank at coarse space lv x.
        /// @param local_cco_ids Bank local CCO id at coarse space lv x.
        __global__ void build_local_cco_lvx(int level,
                                            int cco_num,
                                            ctd::span<const uint32_t> neighbors,
                                            ctd::span<const int> coarse_space_map,
                                            ctd::span<int> cco_num_per_bank,
                                            ctd::span<int> local_cco_ids)
        {

            int tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread id
            int bid = tid / 32;                              // bank id
            int lid = tid % 32;                              // lane id
            if (tid >= cco_num)
            {
                return;
            }

            // Build connectivity mask (including self).
            uint32_t connection = neighbors[tid];
            uint32_t visited = (1u << lid);
            uint32_t to_visit = connection ^ visited;
            while (to_visit)
            {
                int visiting = ctd::countr_one(to_visit);
                connection |= neighbors[visiting + bid * 32];
                visited |= (1u << visiting);
                to_visit = connection ^ visited;
            }

            // Find bank (warp) local connected component (CCO).
            uint32_t cco_lead_lid = ctd::countr_one(connection);
            bool is_lead = (cco_lead_lid == lid);
            uint32_t lead_lanes = __ballot_sync(0xFFFFFFFFu, is_lead);
            uint32_t before_cco_lead_mask = static_cast<uint32_t>((1ull << cco_lead_lid) - 1ull);

            // Write cco info back.
            local_cco_ids[tid] = ctd::popcount(lead_lanes | before_cco_lead_mask);
            if (lid == 0)
            {
                int bank_cco_num = ctd::popcount(lead_lanes);
                cco_num_per_bank[bid] = bank_cco_num;
            }
        };

        /// @brief Build global coarse space level x CCO id.
        /// @param level Target coarse space lv x.
        /// @param local_cco_ids Bank local CCO id at coarse space lv x.
        /// @param cco_num_per_bank Independent CCO number per bank at coarse space lv x.
        /// @param cco_num_per_bank_summed Inclusive sum of cco_num_per_bank.
        /// @param coarse_space_map Global CCO id out.
        __global__ void build_global_cco_lvx(
            int level,
            ctd::span<const int> local_cco_ids,
            ctd::span<const int> cco_num_per_bank,
            ctd::span<const int> cco_num_per_bank_summed,
            ctd::span<int> coarse_space_map)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread id
            int bid = tid / 32;                              // bank id

            if (tid >= local_cco_ids.size())
            {
                return;
            }

            int cco_id = local_cco_ids[tid] + cco_num_per_bank_summed[bid] - cco_num_per_bank[bid];
            coarse_space_map[tid * MAX_COARSE_LEVEL + level] = cco_id;
        }

        /// @brief Build coarse space map.
        ///
        /// Coarse space map id layout:
        /// | node0_lv0 node0_lv1 ... node0_lv_max | node1_lv0 ... node1_lv_max | ... |
        ///
        /// @param row_ptr CSR graph topology (does not include self).
        /// @param cols CSR graph topology (does not include self).
        /// @rt Cuda runtime config.
        /// @warning Will modify cols as side effect!
        CoarseSpace build_coarse_space(ctd::span<const int> row_ptr, ctd::span<int> cols, CudaRuntime rt)
        {
            ctd::array<int, MAX_COARSE_LEVEL> cco_nums;

            int node_num = row_ptr.size() - 1;
            int max_bank_per_level = div_round_upper(node_num, 32);
            auto local_cco_ids =
                cu::make_buffer<int>(rt.stream, rt.mr, node_num, cu::no_init);
            auto cco_num_per_bank =
                cu::make_buffer<int>(rt.stream, rt.mr, max_bank_per_level, cu::no_init);
            auto cco_num_per_bank_summed =
                cu::make_buffer<int>(rt.stream, rt.mr, max_bank_per_level, cu::no_init);
            auto coarse_space_map =
                cu::make_buffer<int>(rt.stream, rt.mr, node_num * MAX_COARSE_LEVEL, cu::no_init);

            // Build coarse map level 0.
            int grid_num = div_round_upper(node_num, 128);
            build_local_cco_lv0<<<grid_num, 128, 0, rt.stream.get()>>>(
                row_ptr, cols, cco_num_per_bank, local_cco_ids);

            size_t cub_tmp_size;
            cub::DeviceScan::InclusiveSum(nullptr,
                                          cub_tmp_size,
                                          cco_num_per_bank.data(),
                                          cco_num_per_bank_summed.data(),
                                          max_bank_per_level,
                                          rt.stream.get());
            auto cub_tmp =
                cu::make_buffer<char>(rt.stream, rt.mr, cub_tmp_size, cu::no_init);
            cub::DeviceScan::InclusiveSum(cub_tmp.data(),
                                          cub_tmp_size,
                                          cco_num_per_bank.data(),
                                          cco_num_per_bank_summed.data(),
                                          max_bank_per_level,
                                          rt.stream.get());

            build_global_cco<<<grid_num, 128, 0, rt.stream.get()>>>(
                0, local_cco_ids, cco_num_per_bank, cco_num_per_bank_summed, coarse_space_map);

            int cco_num = device2host(cco_num_per_bank_summed.data() + max_bank_per_level - 1, rt);
            cco_nums[0] = cco_num;

            // Build coarse map level 1 ... MAX_COARSE_LEVEL-1 recursively.
            auto neighbors = cu::make_buffer<uint32_t>(rt.stream, rt.mr, cco_num, cu::no_init);
            for (int lv = 1; lv < MAX_COARSE_LEVEL; ++lv)
            {
                int grid_num = div_round_upper(node_num, 128);
                build_neighbor_masks_lvx<<<grid_num, 128, 0, rt.stream.get()>>>(
                    lv, coarse_space_map, row_ptr, cols, neighbors);

                grid_num = div_round_upper(cco_num, 128);
                build_local_cco_lvx<<<grid_num, 128, 0, rt.stream.get()>>>(
                    lv, cco_num, neighbors, coarse_space_map, cco_num_per_bank, local_cco_ids);

                int bank_num = div_round_upper(cco_num, 32);
                cub::DeviceScan::InclusiveSum(cub_tmp.data(),
                                              cub_tmp_size,
                                              cco_num_per_bank.data(),
                                              cco_num_per_bank_summed.data(),
                                              bank_num,
                                              rt.stream.get());
                build_global_cco_lvx<<<grid_num, 128, 0, rt.stream.get()>>>(
                    lv, local_cco_ids, cco_num_per_bank, cco_num_per_bank_summed, coarse_space_map);

                cco_num = device2host(cco_num_per_bank_summed.data() + bank_num - 1, rt);
                cco_nums[lv] = cco_num;
            }

            // No need to sync. device2host already sync the stream implicitly.

            CoarseSpace cs;
            cs.cco_nums = cco_nums;
            cs.map = std::move(coarse_space_map);
            return cs;
        }

        __both__ double *index_upper_mat(double *mat, int dim, int i, int j)
        {
            assert(i <= j);
            return mat + (i * dim) - (i * (i + 1) / 2) + j;
        }

        __global__ void fill_coarse_matrices(BSRView mat_in, CoarseMatrices mat_out, ctd::span<const int> coarse_space_map)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread id
            if (tid > mat_in.dim)
            {
                return;
            }

            int block_in_offset = -1;
            for (int i = mat_in.rows[tid]; i < mat_in.rows[tid + 1]; ++i)
            {
                if (mat_in.cols[i] == tid)
                {
                    block_in_offset = mat_in.block_dim * mat_in.block_dim * i;
                }
            }
            if (block_in_offset == -1)
            {
                return;
            }

            for (int lv = 0; lv < MAX_COARSE_LEVEL; ++lv)
            {
                int cco_id = coarse_space_map[i * MAX_COARSE_LEVEL + lv];
                int cmat_id = cco_id / 32;
                int mat_ij_root = cco_id % 32;
                double *mat_in;
                double *mat_out = mat_out.matrix_per_level.data() + cmat_id * mat_out.mat_storage_size;
                for (int bi = 0; bi < mat_in.block_dim; ++bi)
                {
                    for (int bj = bi; bi < mat_in.block_dim; ++bj)
                    {
                        double *out = index_upper_mat(mat_out, mat_in.block_dim, mat_ij_root + bi, mat_ij_root + bj);
                        double val = mat_in.vals[block_in_offset + mat_in.block_dim * i + j];
                        atomicAdd(out, val);
                    }
                }
            }
        };

        __global__ void invert_upper_packed_96x96(double *d_A, int lda)
        {
            // Packed storage: 96 * 97 / 2 = 4656 doubles (37,248 bytes)
            // Fits natively within the default 48 KB shared memory limit per block.
            __shared__ double s_A[4656];

            int tx = threadIdx.x;
            const int N = 96;

            // Load upper triangle into column-major packed shared memory.
            // Mapping 2D (row, col) to 1D packed index: (col * (col + 1)) / 2 + row
            for (int c = tx; c < N; c++)
            {
                s_A[(c * (c + 1)) / 2 + tx] = d_A[tx * lda + c];
            }
            __syncthreads();

            // Phase 1: In-place Cholesky Factorization
            for (int i = 0; i < N; i++)
            {
                int idx_i_i = (i * (i + 1)) / 2 + i;
                if (tx == i)
                {
                    s_A[idx_i_i] = sqrt(s_A[idx_i_i]);
                }
                __syncthreads();

                if (tx > i)
                {
                    s_A[(tx * (tx + 1)) / 2 + i] /= s_A[idx_i_i];
                }
                __syncthreads();

                if (tx > i)
                {
                    for (int k = tx; k < N; k++)
                    {
                        s_A[(k * (k + 1)) / 2 + tx] -=
                            s_A[(tx * (tx + 1)) / 2 + i] * s_A[(k * (k + 1)) / 2 + i];
                    }
                }
                __syncthreads();
            }

            // Phase 2: Invert U in-place using left-to-right column progression.
            // Computes X_ic = -sum(X_ik * U_kc) / U_cc.
            // This strict sequential progression prevents Read-After-Write (RAW) hazards
            // by ensuring thread computations only rely on already-computed X columns.
            for (int c = 0; c < N; c++)
            {
                int idx_c_c = (c * (c + 1)) / 2 + c;
                double U_cc_inv = 1.0 / s_A[idx_c_c];

                if (tx == c)
                {
                    s_A[idx_c_c] = U_cc_inv;
                }
                __syncthreads();

                if (tx < c)
                {
                    double sum = 0.0;
                    for (int k = tx; k < c; k++)
                    {
                        // X_ik * U_kc
                        sum += s_A[(k * (k + 1)) / 2 + tx] * s_A[(c * (c + 1)) / 2 + k];
                    }
                    s_A[(c * (c + 1)) / 2 + tx] = -sum * U_cc_inv;
                }
                __syncthreads();
            }

            // Phase 3: Matrix Multiplication A^-1 = U^-1 * (U^-1)^T
            // Thread tx computes column tx of the final inverse and writes it to global memory.
            for (int i = 0; i <= tx; i++)
            {
                double sum = 0.0;
                for (int k = tx; k < N; k++)
                {
                    // U^-1_ik * U^-1_tx,k
                    sum += s_A[(k * (k + 1)) / 2 + i] * s_A[(k * (k + 1)) / 2 + tx];
                }
                d_A[i * lda + tx] = sum;
                if (i != tx)
                {
                    d_A[tx * lda + i] = sum; // Populate lower triangle to make global memory matrix symmetric
                }
            }
        }

        // Build symmetric upper triangular coarse space matrices.
        cu::device_buffer<double> build_sym_coarse_matrices(const CoarseSpace &cs, BSRView mat, CudaRuntime rt)
        {
            int mat_n = 32 * mat.block_dim;
            int sym_mat_size = (mat_n * (mat_n - 1)) / 2;
            int total_sym_mat_size = 0;
            ctd::array<int, 4> lv_mat_num;
            for (int i = 0; i < MAX_COARSE_LEVEL; ++i)
            {
                lv_mat_num[i] = div_round_upper(cs.cco_nums[i], 32);

                total_sym_mat_size += lv_mat_num[i] * sym_mat_size;
            }

            CoarseMatrices cmats;
            cmats.mat_dim = mat_n;
            cmats.mat_storage_size = sym_mat_size;
            auto buf = cu::make_buffer<double>(rt.stream, rt.mr, total_sym_mat_size, 0);

            total_sym_mat_size = 0;
            for (int i = 0; i < MAX_COARSE_LEVEL; ++i)
            {
                cmats.matrix_per_level[i] =
                    ctd::span<double>{buf.data() + total_sym_mat_size, lv_mat_num[i] * sym_mat_size};
                total_sym_mat_size += lv_mat_num[i] * sym_mat_size;
            }

            // populate
            int grid_num = div_round_upper(mat.dim, 128);
            fill_coarse_matrices<<<grid_num, 128, 0, rt.stream.get()>>>(mat, cmats, cs.map);

            auto pad_trailing = [cmats] __device__(int lv) {
                int offset = cmats.matrix_per_level[i].size() - mat_storage_size - 1;
                double *mat = cmats.matrix_per_level[i].data() + offset;
                for (int i = 0; i < cmats.mat_dim; ++i)
                {
                    double *diag = index_upper_mat(mat, cmats.mat_dim, i, i);
                    if (*diag == 0.0)
                    {
                        *diag = 1.0;
                    }
                }
            };
            cub::DeviceFor::Bulk(4, pad_trailing, rt.stream.get());

            // Compute inverse
        }
    } // namespace

} // namespace polysolve::linear::mas

#endif

namespace polysolve::linear::mas
{
    void MASPreconditioner::factorize(const BSRMatrix &A, CudaRuntime)
    {
        BSRView view = A.view();
        vector_size_ = view.dim * view.block_dim;
        initialized_ = true;
    }

    void MASPreconditioner::apply(
        ctd::span<const double> r,
        ctd::span<double> z,
        CudaRuntime rt)
    {
        if (!initialized_)
        {
            throw std::runtime_error("[CudaPCG] MASPreconditioner is not initialized.");
        }
        if (r.size() != z.size() || r.size() != vector_size_)
        {
            throw std::runtime_error("[CudaPCG] Invalid vector size for MAS preconditioner.");
        }

        cudaMemcpyAsync(
            z.data(),
            r.data(),
            r.size() * sizeof(double),
            cudaMemcpyDeviceToDevice,
            rt.stream.get());
    }
} // namespace polysolve::linear::mas
