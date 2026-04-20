#include <polysolve/linear/mas_utils/MASPreconditioner.hpp>
#include <polysolve/linear/mas_utils/MASPreconditionerTest.cuh>

#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif

#include <cub/cub.cuh>
#include <cuda/algorithm>
#include <cuda/std/array>
#include <cuda/std/bit>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>

namespace polysolve::linear::mas
{
    namespace
    {
        using clock = std::chrono::steady_clock;

        double elapsed_seconds(const std::chrono::time_point<clock> &begin)
        {
            return std::chrono::duration<double>(clock::now() - begin).count();
        }

        constexpr int BANK_SIZE = 32;
        constexpr int MAX_COARSE_LEVEL = MAS_MAX_COARSE_LEVEL;
        constexpr int LEVEL_COUNT = MAS_LEVEL_COUNT;

        struct HostPaddedTopology
        {
            std::vector<int> real_to_padded;
            std::vector<int> padded_to_real;
            std::vector<int> row_ptr;
            std::vector<int> cols;
            int node_num = 0;
            int padded_node_num = 0;
        };

        struct CoarseMatricesView
        {
            int mat_dim;
            int mat_storage_size;
            ctd::array<ctd::span<double>, LEVEL_COUNT> matrix_per_level;
        };

        HostPaddedTopology build_padded_topology(TopologyView topo, ctd::span<const int> part_offsets)
        {
            int node_num = topo.row_ptr.size() - 1;
            int part_num = part_offsets.size() - 1;

            std::vector<int> padded_offsets(part_num + 1, 0);
            for (int part = 0; part < part_num; ++part)
            {
                int part_size = part_offsets[part + 1] - part_offsets[part];
                int padded_size = div_round_upper(part_size, BANK_SIZE) * BANK_SIZE;
                padded_offsets[part + 1] = padded_offsets[part] + padded_size;
            }

            HostPaddedTopology out;
            out.real_to_padded.assign(node_num, -1);
            out.padded_to_real.assign(node_num, -1);
            out.row_ptr.assign(padded_offsets[part_num] + 1, 0);
            out.node_num = node_num;
            out.padded_node_num = padded_offsets[part_num];

            std::vector<int> padded_to_real(out.padded_node_num, -1);

            for (int part = 0; part < part_num; ++part)
            {
                int part_begin = part_offsets[part];
                int part_end = part_offsets[part + 1];
                int padded_begin = padded_offsets[part];
                int padded_end = padded_offsets[part + 1];

                for (int padded_id = padded_begin; padded_id < padded_end; ++padded_id)
                {
                    int local_id = padded_id - padded_begin;
                    if (part_begin + local_id < part_end)
                    {
                        int real_id = part_begin + local_id;
                        out.real_to_padded[real_id] = padded_id;
                        out.padded_to_real[real_id] = padded_id;
                        padded_to_real[padded_id] = real_id;
                        out.row_ptr[padded_id + 1] = topo.row_ptr[real_id + 1] - topo.row_ptr[real_id];
                    }
                }
            }

            for (int i = 0; i < out.padded_node_num; ++i)
            {
                out.row_ptr[i + 1] += out.row_ptr[i];
            }
            out.cols.resize(out.row_ptr.back());

            for (int padded_id = 0; padded_id < out.padded_node_num; ++padded_id)
            {
                int real_id = padded_to_real[padded_id];
                if (real_id == -1)
                {
                    continue;
                }

                int dst = out.row_ptr[padded_id];
                for (int n = topo.row_ptr[real_id]; n < topo.row_ptr[real_id + 1]; ++n)
                {
                    out.cols[dst] = out.real_to_padded[topo.cols[n]];
                    ++dst;
                }
            }

            return out;
        }

        CoarseMatricesView make_coarse_matrices_view(CoarseMatrices &mats)
        {
            CoarseMatricesView out;
            out.mat_dim = mats.mat_dim;
            out.mat_storage_size = mats.mat_storage_size;

            int scalar_offset = 0;
            for (int i = 0; i < LEVEL_COUNT; ++i)
            {
                int level_size = mats.matrix_counts[i] * mats.mat_storage_size;
                out.matrix_per_level[i] =
                    ctd::span<double>(mats.data->data() + scalar_offset, level_size);
                scalar_offset += level_size;
            }

            return out;
        }

        /// @brief Get coarse space CCO id.
        /// @param map Coarse space map.
        /// @param vid Vertex id.
        /// @param level Coarse space level. Start at lv 0.
        /// @return CCO id.
        __both__ int get_coarse_space_id(ctd::span<const int> map, int vid, int level)
        {
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
            int wid = threadIdx.x / BANK_SIZE;        // block local warp id
            int lid = threadIdx.x % BANK_SIZE;        // lane id
            int bid = tid / BANK_SIZE;                // bank id

            int node_num = row_ptr.size() - 1;
            if (tid >= node_num)
            {
                return;
            }

            // Build neighbor mask (including self) and collapse topology.
            int row_begin = row_ptr[tid];
            int row_end = row_ptr[tid + 1];
            uint32_t neighbor = (1u << lid);
            int out_of_bank_neighbor_count = 0;
            for (int n = row_begin; n < row_end; ++n)
            {
                int neighbor_vid = cols[n];
                int neighbor_bid = neighbor_vid / BANK_SIZE;
                if (bid == neighbor_bid)
                {
                    neighbor |= (1u << (neighbor_vid % BANK_SIZE));
                }
                else
                {
                    // Compact topology to include out of bank neighbors exclusively. So we dont
                    // need additional filtering when building future coarse spaces.
                    cols[row_begin + out_of_bank_neighbor_count] = neighbor_vid;
                    ++out_of_bank_neighbor_count;
                }
            }
            if (row_begin + out_of_bank_neighbor_count < row_end)
            {
                // -1 denotes neighbor list ending.
                cols[row_begin + out_of_bank_neighbor_count] = -1;
            }
            neighbors[btid] = neighbor;
            __syncwarp();

            // Build connectivity mask (including self).
            uint32_t connection = neighbor;
            uint32_t visited = (1u << lid);
            uint32_t to_visit = connection ^ visited;
            while (to_visit)
            {
                int visiting = ctd::countr_zero(to_visit);
                connection |= neighbors[visiting + wid * BANK_SIZE];
                visited |= (1u << visiting);
                to_visit = connection ^ visited;
            }

            // Find bank (warp) local connected component (CCO).
            uint32_t cco_lead_lid = ctd::countr_zero(connection);
            bool is_lead = (cco_lead_lid == lid);
            uint32_t lead_lanes = __ballot_sync(0xFFFFFFFFu, is_lead);
            uint32_t before_cco_lead_mask = static_cast<uint32_t>((1ull << cco_lead_lid) - 1ull);

            // Write cco info back.
            local_cco_ids[tid] = ctd::popcount(lead_lanes & before_cco_lead_mask);
            if (lid == 0)
            {
                cco_num_per_bank[bid] = ctd::popcount(lead_lanes);
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
            int row_begin = row_ptr[tid];
            int row_end = row_ptr[tid + 1];
            int cco_id = get_coarse_space_id(coarse_space_map, tid, level - 1);
            uint32_t neighbor = (1u << (cco_id % BANK_SIZE));
            int out_of_bank_neighbor_count = 0;
            for (int n = row_begin; n < row_end; ++n)
            {
                int neighbor_vid = cols[n];
                if (neighbor_vid == -1)
                {
                    break;
                }

                int neighbor_cco_id = get_coarse_space_id(coarse_space_map, neighbor_vid, level - 1);
                if (cco_id / BANK_SIZE == neighbor_cco_id / BANK_SIZE)
                {
                    neighbor |= (1u << (neighbor_cco_id % BANK_SIZE));
                }
                else
                {
                    cols[row_begin + out_of_bank_neighbor_count] = neighbor_vid;
                    ++out_of_bank_neighbor_count;
                }
            }
            if (row_begin + out_of_bank_neighbor_count < row_end)
            {
                cols[row_begin + out_of_bank_neighbor_count] = -1;
            }

            atomicOr(neighbors.data() + cco_id, neighbor);
        }

        /// @brief Build local CCO mapping from coarse space level x-1 -> coarse space level x.
        /// @param cco_num Total CCO number at coarse space lv x-1.
        /// @param cco_num_per_bank Independent CCO number per bank at coarse space lv x.
        /// @param local_cco_ids Bank local CCO id at coarse space lv x.
        __global__ void build_local_cco_lvx(int cco_num,
                                            ctd::span<const uint32_t> neighbors,
                                            ctd::span<int> cco_num_per_bank,
                                            ctd::span<int> local_cco_ids)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread id
            int bid = tid / BANK_SIZE;                       // bank id
            int lid = tid % BANK_SIZE;                       // lane id
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
                int visiting = ctd::countr_zero(to_visit);
                connection |= neighbors[visiting + bid * BANK_SIZE];
                visited |= (1u << visiting);
                to_visit = connection ^ visited;
            }

            // Find bank (warp) local connected component (CCO).
            uint32_t cco_lead_lid = ctd::countr_zero(connection);
            bool is_lead = (cco_lead_lid == lid);
            uint32_t lead_lanes = __ballot_sync(0xFFFFFFFFu, is_lead);
            uint32_t before_cco_lead_mask = static_cast<uint32_t>((1ull << cco_lead_lid) - 1ull);

            // Write cco info back.
            local_cco_ids[tid] = ctd::popcount(lead_lanes & before_cco_lead_mask);
            if (lid == 0)
            {
                cco_num_per_bank[bid] = ctd::popcount(lead_lanes);
            }
        }

        __global__ void build_global_cco_ids(ctd::span<const int> local_cco_ids,
                                             ctd::span<const int> cco_num_per_bank,
                                             ctd::span<const int> cco_num_per_bank_summed,
                                             ctd::span<int> global_cco_ids)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread id
            int bid = tid / BANK_SIZE;                       // bank id
            if (tid >= local_cco_ids.size())
            {
                return;
            }

            global_cco_ids[tid] =
                local_cco_ids[tid] + cco_num_per_bank_summed[bid] - cco_num_per_bank[bid];
        }

        /// @brief Build global coarse space level 0 CCO id.
        /// @param local_cco_ids Bank local CCO id at coarse space lv 0.
        /// @param cco_num_per_bank Independent CCO number per bank at coarse space lv 0.
        /// @param cco_num_per_bank_summed Inclusive sum of cco_num_per_bank.
        /// @param coarse_space_map Global CCO id out.
        __global__ void build_global_cco_lv0(ctd::span<const int> local_cco_ids,
                                             ctd::span<const int> cco_num_per_bank,
                                             ctd::span<const int> cco_num_per_bank_summed,
                                             ctd::span<int> coarse_space_map)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread id
            int bid = tid / BANK_SIZE;                       // bank id

            if (tid >= local_cco_ids.size())
            {
                return;
            }

            int cco_id = local_cco_ids[tid] + cco_num_per_bank_summed[bid] - cco_num_per_bank[bid];
            coarse_space_map[tid * MAX_COARSE_LEVEL] = cco_id;
        }

        /// @brief Build global coarse space level x CCO id.
        /// @param level Target coarse space lv x.
        /// @param coarse_space_map Coarse space map.
        /// @param global_cco_ids Global CCO id of previous coarse space components.
        __global__ void build_global_cco_lvx(int level,
                                             ctd::span<int> coarse_space_map,
                                             ctd::span<const int> global_cco_ids)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread id
            int node_num = coarse_space_map.size() / MAX_COARSE_LEVEL;
            if (tid >= node_num)
            {
                return;
            }

            int prev_cco_id = get_coarse_space_id(coarse_space_map, tid, level - 1);
            coarse_space_map[tid * MAX_COARSE_LEVEL + level] = global_cco_ids[prev_cco_id];
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
            std::array<int, MAX_COARSE_LEVEL> cco_nums{};

            int node_num = row_ptr.size() - 1;
            int max_bank_per_level = div_round_upper(node_num, BANK_SIZE);
            auto local_cco_ids =
                cu::make_buffer<int>(rt.stream, rt.mr, node_num, cu::no_init);
            auto cco_num_per_bank =
                cu::make_buffer<int>(rt.stream, rt.mr, max_bank_per_level, cu::no_init);
            auto cco_num_per_bank_summed =
                cu::make_buffer<int>(rt.stream, rt.mr, max_bank_per_level, cu::no_init);
            auto global_cco_ids =
                cu::make_buffer<int>(rt.stream, rt.mr, node_num, cu::no_init);
            auto coarse_space_map =
                cu::make_buffer<int>(rt.stream, rt.mr, node_num * MAX_COARSE_LEVEL, cu::no_init);

            // Build coarse map level 0.
            int grid_num = div_round_upper(node_num, 128);
            build_local_cco_lv0<<<grid_num, 128, 0, rt.stream.get()>>>(
                row_ptr, cols, cco_num_per_bank, local_cco_ids);

            size_t cub_tmp_size = 0;
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

            build_global_cco_lv0<<<grid_num, 128, 0, rt.stream.get()>>>(
                local_cco_ids, cco_num_per_bank, cco_num_per_bank_summed, coarse_space_map);

            int cco_num = device2host(cco_num_per_bank_summed.data() + max_bank_per_level - 1, rt);
            cco_nums[0] = cco_num;
            int level_num = 1;

            // Build coarse map level 1 ... MAX_COARSE_LEVEL-1 recursively.
            auto neighbors = cu::make_buffer<uint32_t>(rt.stream, rt.mr, node_num, cu::no_init);
            for (int lv = 1; lv < MAX_COARSE_LEVEL; ++lv)
            {
                cudaMemsetAsync(neighbors.data(), 0, cco_num * sizeof(uint32_t), rt.stream.get());

                grid_num = div_round_upper(node_num, 128);
                build_neighbor_masks_lvx<<<grid_num, 128, 0, rt.stream.get()>>>(
                    lv, coarse_space_map, row_ptr, cols, ctd::span<uint32_t>(neighbors.data(), cco_num));

                int bank_num = div_round_upper(cco_num, BANK_SIZE);
                grid_num = div_round_upper(cco_num, 128);
                build_local_cco_lvx<<<grid_num, 128, 0, rt.stream.get()>>>(
                    cco_num,
                    ctd::span<const uint32_t>(neighbors.data(), cco_num),
                    ctd::span<int>(cco_num_per_bank.data(), bank_num),
                    ctd::span<int>(local_cco_ids.data(), cco_num));

                cub::DeviceScan::InclusiveSum(cub_tmp.data(),
                                              cub_tmp_size,
                                              cco_num_per_bank.data(),
                                              cco_num_per_bank_summed.data(),
                                              bank_num,
                                              rt.stream.get());
                build_global_cco_ids<<<grid_num, 128, 0, rt.stream.get()>>>(
                    ctd::span<const int>(local_cco_ids.data(), cco_num),
                    ctd::span<const int>(cco_num_per_bank.data(), bank_num),
                    ctd::span<const int>(cco_num_per_bank_summed.data(), bank_num),
                    ctd::span<int>(global_cco_ids.data(), cco_num));
                build_global_cco_lvx<<<div_round_upper(node_num, 128), 128, 0, rt.stream.get()>>>(
                    lv, coarse_space_map, ctd::span<const int>(global_cco_ids.data(), cco_num));

                int next_cco_num = device2host(cco_num_per_bank_summed.data() + bank_num - 1, rt);
                if (next_cco_num == cco_num)
                {
                    break;
                }

                cco_num = next_cco_num;
                cco_nums[lv] = cco_num;
                level_num = lv + 1;
            }

            CoarseSpace cs;
            cs.cco_nums = cco_nums;
            cs.level_num = level_num;
            cs.map = std::move(coarse_space_map);
            return cs;
        }

        __both__ int index_upper_mat(int dim, int i, int j)
        {
            if (i > j)
            {
                int tmp = i;
                i = j;
                j = tmp;
            }
            return i * dim - (i * (i + 1) / 2) + j;
        }

        __global__ void fill_coarse_matrices(BSRView mat_in,
                                             CoarseMatricesView mat_out,
                                             ctd::span<const int> coarse_space_map,
                                             ctd::span<const int> real_to_padded,
                                             int coarse_level_num)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x; // global thread id
            if (tid >= mat_in.dim)
            {
                return;
            }

            int padded_i = real_to_padded[tid];
            int block_size = mat_in.block_dim * mat_in.block_dim;
            for (int nz = mat_in.rows[tid]; nz < mat_in.rows[tid + 1]; ++nz)
            {
                int j = mat_in.cols[nz];
                if (tid > j)
                {
                    continue;
                }

                int padded_j = real_to_padded[j];
                int block_offset = nz * block_size;
                if (padded_i / BANK_SIZE == padded_j / BANK_SIZE)
                {
                    int mat_id = padded_i / BANK_SIZE;
                    int scalar_i_root = (padded_i % BANK_SIZE) * mat_in.block_dim;
                    int scalar_j_root = (padded_j % BANK_SIZE) * mat_in.block_dim;
                    double *mat = mat_out.matrix_per_level[0].data() + mat_id * mat_out.mat_storage_size;
                    for (int bi = 0; bi < mat_in.block_dim; ++bi)
                    {
                        for (int bj = 0; bj < mat_in.block_dim; ++bj)
                        {
                            int row = scalar_i_root + bi;
                            int col = scalar_j_root + bj;
                            if (row > col)
                            {
                                continue;
                            }

                            if (tid == j && bi > bj)
                            {
                                continue;
                            }

                            double val = mat_in.vals[block_offset + bi * mat_in.block_dim + bj];
                            atomicAdd(mat + index_upper_mat(mat_out.mat_dim, row, col), val);
                        }
                    }
                }
                for (int lv = 0; lv < coarse_level_num; ++lv)
                {
                    int cco_i = coarse_space_map[padded_i * MAX_COARSE_LEVEL + lv];
                    int cco_j = coarse_space_map[padded_j * MAX_COARSE_LEVEL + lv];
                    if (cco_i / BANK_SIZE != cco_j / BANK_SIZE)
                    {
                        continue;
                    }

                    int mat_id = cco_i / BANK_SIZE;
                    int scalar_i_root = (cco_i % BANK_SIZE) * mat_in.block_dim;
                    int scalar_j_root = (cco_j % BANK_SIZE) * mat_in.block_dim;
                    double *mat = mat_out.matrix_per_level[lv + 1].data() + mat_id * mat_out.mat_storage_size;
                    for (int bi = 0; bi < mat_in.block_dim; ++bi)
                    {
                        for (int bj = 0; bj < mat_in.block_dim; ++bj)
                        {
                            int row = scalar_i_root + bi;
                            int col = scalar_j_root + bj;
                            if (row > col)
                            {
                                continue;
                            }

                            if (tid == j && bi > bj)
                            {
                                continue;
                            }

                            double val = mat_in.vals[block_offset + bi * mat_in.block_dim + bj];
                            atomicAdd(mat + index_upper_mat(mat_out.mat_dim, row, col), val);
                        }
                    }
                }
            }
        }

        __global__ void gather_multi_level_r(
            ctd::span<const double> r,
            ctd::span<double> multi_level_r,
            ctd::span<const int> real_to_padded,
            ctd::span<const int> coarse_space_map,
            ctd::span<const int> level_offsets,
            int block_dim,
            int coarse_level_num)
        {
            int real_id = blockDim.x * blockIdx.x + threadIdx.x;
            if (real_id >= real_to_padded.size())
            {
                return;
            }

            int padded_id = real_to_padded[real_id];
            int src_root = real_id * block_dim;
            int fine_root = padded_id * block_dim;
            for (int comp = 0; comp < block_dim; ++comp)
            {
                atomicAdd(multi_level_r.data() + fine_root + comp, r[src_root + comp]);
            }
            for (int lv = 0; lv < coarse_level_num; ++lv)
            {
                int cco_id = coarse_space_map[padded_id * MAX_COARSE_LEVEL + lv];
                int dst_root = (level_offsets[lv + 1] + cco_id) * block_dim;
                for (int comp = 0; comp < block_dim; ++comp)
                {
                    atomicAdd(multi_level_r.data() + dst_root + comp, r[src_root + comp]);
                }
            }
        }

        template <int N, int BLOCK>
        __global__ void symv_upper_packed(const double *__restrict__ A_upper,
                                          const double *__restrict__ x,
                                          double *__restrict__ y,
                                          int num_mats)
        {
            int mat = blockIdx.x;
            if (mat >= num_mats)
            {
                return;
            }

            int row = threadIdx.x;
            constexpr int L = N * (N + 1) / 2;
            const double *Amat = A_upper + static_cast<size_t>(mat) * L;

            __shared__ double sx[N];
            __shared__ double sA[L];

            for (int i = threadIdx.x; i < N; i += BLOCK)
            {
                sx[i] = x[static_cast<size_t>(mat) * N + i];
            }
            for (int k = threadIdx.x; k < L; k += BLOCK)
            {
                sA[k] = Amat[k];
            }

            __syncthreads();

            if (row >= N)
            {
                return;
            }

            double sum = 0.0;
            for (int col = 0; col < N; ++col)
            {
                int r = row <= col ? row : col;
                int c = row <= col ? col : row;
                sum += sA[index_upper_mat(N, r, c)] * sx[col];
            }

            y[static_cast<size_t>(mat) * N + row] = sum;
        }

        void apply_packed_matrices(const CoarseMatrices &mats,
                                   ctd::span<const double> x,
                                   ctd::span<double> y,
                                   int block_dim,
                                   CudaRuntime rt)
        {
            int mat_num = mats.total_matrix_num;
            if (mat_num == 0)
            {
                return;
            }

            if (block_dim == 1)
            {
                symv_upper_packed<32, 32><<<mat_num, 32, 0, rt.stream.get()>>>(
                    mats.data->data(),
                    x.data(),
                    y.data(),
                    mat_num);
                return;
            }
            if (block_dim == 2)
            {
                symv_upper_packed<64, 64><<<mat_num, 64, 0, rt.stream.get()>>>(
                    mats.data->data(),
                    x.data(),
                    y.data(),
                    mat_num);
                return;
            }
            if (block_dim == 3)
            {
                symv_upper_packed<96, 96><<<mat_num, 96, 0, rt.stream.get()>>>(
                    mats.data->data(),
                    x.data(),
                    y.data(),
                    mat_num);
                return;
            }

            throw std::runtime_error("[CudaPCG] MAS only supports block size 1, 2, or 3.");
        }

        __global__ void collect_multi_level_z(
            ctd::span<const double> multi_level_z,
            ctd::span<double> z,
            ctd::span<const int> real_to_padded,
            ctd::span<const int> coarse_space_map,
            ctd::span<const int> level_offsets,
            int block_dim,
            int coarse_level_num)
        {
            int real_id = blockDim.x * blockIdx.x + threadIdx.x;
            if (real_id >= real_to_padded.size())
            {
                return;
            }

            int padded_id = real_to_padded[real_id];
            int dst_root = real_id * block_dim;
            for (int comp = 0; comp < block_dim; ++comp)
            {
                z[dst_root + comp] = multi_level_z[padded_id * block_dim + comp];
            }

            for (int lv = 0; lv < coarse_level_num; ++lv)
            {
                int cco_id = coarse_space_map[padded_id * MAX_COARSE_LEVEL + lv];
                int src_root = (level_offsets[lv + 1] + cco_id) * block_dim;
                for (int comp = 0; comp < block_dim; ++comp)
                {
                    z[dst_root + comp] += multi_level_z[src_root + comp];
                }
            }
        }

        __global__ void pad_zero_diagonal(double *mats, int mat_num, int mat_dim, int mat_storage_size)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            int total_diag = mat_num * mat_dim;
            if (tid >= total_diag)
            {
                return;
            }

            int mat_id = tid / mat_dim;
            int row = tid % mat_dim;
            double *mat = mats + mat_id * mat_storage_size;
            double *diag = mat + index_upper_mat(mat_dim, row, row);
            if (*diag == 0.0)
            {
                *diag = 1.0;
            }
        }

        template <int N>
        __global__ void batched_invert_upper_packed(double *d_matrices, int mat_num)
        {
            int mat_idx = blockIdx.x;
            if (mat_idx >= mat_num)
            {
                return;
            }

            constexpr int STORAGE = N * (N + 1) / 2;
            double *d_A = d_matrices + mat_idx * STORAGE;

            __shared__ double s_A[STORAGE];
            int tx = threadIdx.x;

            for (int i = tx; i < STORAGE; i += N)
            {
                s_A[i] = d_A[i];
            }
            __syncthreads();

            // Phase 1: In-place Cholesky Factorization (A = U^T U)
            for (int i = 0; i < N; ++i)
            {
                if (tx == i)
                {
                    s_A[index_upper_mat(N, i, i)] = sqrt(s_A[index_upper_mat(N, i, i)]);
                }
                __syncthreads();

                if (tx > i)
                {
                    s_A[index_upper_mat(N, i, tx)] /= s_A[index_upper_mat(N, i, i)];
                }
                __syncthreads();

                if (tx > i)
                {
                    double U_i_tx = s_A[index_upper_mat(N, i, tx)];
                    for (int c = tx; c < N; ++c)
                    {
                        s_A[index_upper_mat(N, tx, c)] -= U_i_tx * s_A[index_upper_mat(N, i, c)];
                    }
                }
                __syncthreads();
            }

            // Phase 2: Invert U in-place (U^-1)
            for (int c = 0; c < N; ++c)
            {
                double inv_Ucc = 1.0 / s_A[index_upper_mat(N, c, c)];
                double new_val = 0.0;

                if (tx < c)
                {
                    double sum = 0.0;
                    for (int k = tx; k < c; ++k)
                    {
                        sum += s_A[index_upper_mat(N, tx, k)] * s_A[index_upper_mat(N, k, c)];
                    }
                    new_val = -sum * inv_Ucc;
                }
                __syncthreads();

                if (tx < c)
                {
                    s_A[index_upper_mat(N, tx, c)] = new_val;
                }
                else if (tx == c)
                {
                    s_A[index_upper_mat(N, c, c)] = inv_Ucc;
                }
                __syncthreads();
            }

            // Phase 3: Matrix Multiplication A^-1 = U^-1 * U^-T
            for (int c = 0; c < N; ++c)
            {
                double new_val = 0.0;

                if (tx <= c)
                {
                    double sum = 0.0;
                    for (int k = c; k < N; ++k)
                    {
                        sum += s_A[index_upper_mat(N, tx, k)] * s_A[index_upper_mat(N, c, k)];
                    }
                    new_val = sum;
                }
                __syncthreads();

                if (tx <= c)
                {
                    s_A[index_upper_mat(N, tx, c)] = new_val;
                }
                __syncthreads();
            }

            for (int i = tx; i < STORAGE; i += N)
            {
                d_A[i] = s_A[i];
            }
        }

        void invert_packed_matrices(double *mats, int mat_num, int block_dim, CudaRuntime rt)
        {
            if (mat_num == 0)
            {
                return;
            }

            if (block_dim == 1)
            {
                batched_invert_upper_packed<32><<<mat_num, 32, 0, rt.stream.get()>>>(mats, mat_num);
                return;
            }
            if (block_dim == 2)
            {
                batched_invert_upper_packed<64><<<mat_num, 64, 0, rt.stream.get()>>>(mats, mat_num);
                return;
            }
            if (block_dim == 3)
            {
                batched_invert_upper_packed<96><<<mat_num, 96, 0, rt.stream.get()>>>(mats, mat_num);
                return;
            }

            throw std::runtime_error("[CudaPCG] MAS only supports block size 1, 2, or 3.");
        }

        // Build symmetric upper triangular coarse space matrices.
        CoarseMatrices build_sym_coarse_matrices(const CoarseSpace &cs,
                                                 BSRView mat,
                                                 ctd::span<const int> real_to_padded,
                                                 int padded_node_num,
                                                 CudaRuntime rt)
        {
            CoarseMatrices out;
            out.mat_dim = BANK_SIZE * mat.block_dim;
            out.mat_storage_size = out.mat_dim * (out.mat_dim + 1) / 2;
            out.total_matrix_num = 0;
            out.matrix_offsets[0] = 0;
            out.matrix_counts[0] = div_round_upper(padded_node_num, BANK_SIZE);
            out.total_matrix_num += out.matrix_counts[0];
            for (int i = 0; i < cs.level_num; ++i)
            {
                out.matrix_offsets[i + 1] = out.total_matrix_num;
                out.matrix_counts[i + 1] = div_round_upper(cs.cco_nums[i], BANK_SIZE);
                out.total_matrix_num += out.matrix_counts[i + 1];
            }

            out.data = cu::make_buffer<double>(
                rt.stream,
                rt.mr,
                std::max(out.total_matrix_num * out.mat_storage_size, 1),
                0.0);
            CoarseMatricesView view = make_coarse_matrices_view(out);

            int grid_num = div_round_upper(mat.dim, 128);
            fill_coarse_matrices<<<grid_num, 128, 0, rt.stream.get()>>>(
                mat, view, *cs.map, real_to_padded, cs.level_num);

            int total_diag = out.total_matrix_num * out.mat_dim;
            pad_zero_diagonal<<<div_round_upper(total_diag, 128), 128, 0, rt.stream.get()>>>(
                out.data->data(), out.total_matrix_num, out.mat_dim, out.mat_storage_size);

            invert_packed_matrices(out.data->data(), out.total_matrix_num, mat.block_dim, rt);
            return out;
        }
    } // namespace

    void MASPreconditioner::factorize(const BSRMatrix &A,
                                      ctd::span<const int> part_offsets,
                                      CudaRuntime rt)
    {
        if (part_offsets.size() < 2)
        {
            throw std::runtime_error("[CudaPCG] Invalid MAS partition offsets.");
        }

        BSRView view = A.view();
        if (view.block_dim < 1 || view.block_dim > 3)
        {
            throw std::runtime_error("[CudaPCG] MAS only supports block size 1, 2, or 3.");
        }

        initialized_ = false;
        block_dim_ = view.block_dim;
        vector_size_ = view.dim * view.block_dim;

        auto total_begin = clock::now();
        auto phase_begin = clock::now();
        HostPaddedTopology topo = build_padded_topology(A.host_topology_view(), part_offsets);
        SPDLOG_TRACE("CUDA_PCG MAS: build_padded_topology {:.6f}s", elapsed_seconds(phase_begin));

        padded_vector_size_ = topo.padded_node_num * block_dim_;
        int topo_col_size = topo.cols.empty() ? 1 : topo.cols.size();

        phase_begin = clock::now();
        padded_topology_.node_num = topo.node_num;
        padded_topology_.padded_node_num = topo.padded_node_num;
        padded_topology_.real_to_padded =
            cu::make_buffer<int>(rt.stream, rt.mr, topo.real_to_padded.size(), cu::no_init);
        padded_topology_.padded_to_real =
            cu::make_buffer<int>(rt.stream, rt.mr, topo.padded_to_real.size(), cu::no_init);
        padded_topology_.rows =
            cu::make_buffer<int>(rt.stream, rt.mr, topo.row_ptr.size(), cu::no_init);
        padded_topology_.cols =
            cu::make_buffer<int>(rt.stream, rt.mr, topo_col_size, cu::no_init);

        cu::copy_bytes(rt.stream, topo.real_to_padded, *padded_topology_.real_to_padded);
        cu::copy_bytes(rt.stream, topo.padded_to_real, *padded_topology_.padded_to_real);
        cu::copy_bytes(rt.stream, topo.row_ptr, *padded_topology_.rows);
        if (!topo.cols.empty())
        {
            cu::copy_bytes(
                rt.stream,
                topo.cols,
                ctd::span<int>(padded_topology_.cols->data(), topo.cols.size()));
        }
        rt.stream.sync();
        SPDLOG_TRACE("CUDA_PCG MAS: copy_padded_topology {:.6f}s", elapsed_seconds(phase_begin));

        phase_begin = clock::now();
        coarse_space_ = build_coarse_space(
            ctd::span<const int>(padded_topology_.rows->data(), topo.row_ptr.size()),
            ctd::span<int>(padded_topology_.cols->data(), topo.cols.size()),
            rt);
        rt.stream.sync();
        SPDLOG_TRACE("CUDA_PCG MAS: build_coarse_space {:.6f}s", elapsed_seconds(phase_begin));

        phase_begin = clock::now();
        coarse_matrices_ = build_sym_coarse_matrices(
            coarse_space_,
            view,
            ctd::span<const int>(
                padded_topology_.real_to_padded->data(),
                topo.real_to_padded.size()),
            padded_topology_.padded_node_num,
            rt);
        rt.stream.sync();
        SPDLOG_TRACE("CUDA_PCG MAS: build_coarse_matrices {:.6f}s", elapsed_seconds(phase_begin));

        phase_begin = clock::now();
        workspace_.level_offsets[0] = 0;
        workspace_.level_sizes[0] = padded_topology_.padded_node_num;
        workspace_.total_level_nodes = padded_topology_.padded_node_num;
        for (int i = 0; i < coarse_space_.level_num; ++i)
        {
            workspace_.level_offsets[i + 1] = workspace_.total_level_nodes;
            workspace_.level_sizes[i + 1] = coarse_matrices_.matrix_counts[i + 1] * BANK_SIZE;
            workspace_.total_level_nodes += workspace_.level_sizes[i + 1];
        }
        int total_level_scalars = std::max(workspace_.total_level_nodes * block_dim_, 1);
        workspace_.multi_level_r =
            cu::make_buffer<double>(rt.stream, rt.mr, total_level_scalars, 0.0);
        workspace_.multi_level_z =
            cu::make_buffer<double>(rt.stream, rt.mr, total_level_scalars, 0.0);
        rt.stream.sync();
        SPDLOG_TRACE("CUDA_PCG MAS: build_workspace {:.6f}s", elapsed_seconds(phase_begin));
        SPDLOG_TRACE("CUDA_PCG MAS: factorize_total {:.6f}s", elapsed_seconds(total_begin));

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
        if (!workspace_.multi_level_r || !workspace_.multi_level_z)
        {
            throw std::runtime_error("[CudaPCG] MAS workspace is not initialized.");
        }

        int total_level_scalars = workspace_.total_level_nodes * block_dim_;
        cudaMemsetAsync(
            workspace_.multi_level_r->data(),
            0,
            total_level_scalars * sizeof(double),
            rt.stream.get());
        cudaMemsetAsync(
            workspace_.multi_level_z->data(),
            0,
            total_level_scalars * sizeof(double),
            rt.stream.get());

        auto level_offsets =
            ctd::span<const int>(workspace_.level_offsets.data(), LEVEL_COUNT);
        int grid_num = div_round_upper(padded_topology_.node_num, 128);
        gather_multi_level_r<<<grid_num, 128, 0, rt.stream.get()>>>(
            r,
            *workspace_.multi_level_r,
            ctd::span<const int>(
                padded_topology_.real_to_padded->data(),
                padded_topology_.node_num),
            *coarse_space_.map,
            level_offsets,
            block_dim_,
            coarse_space_.level_num);

        apply_packed_matrices(
            coarse_matrices_,
            ctd::span<const double>(workspace_.multi_level_r->data(), total_level_scalars),
            ctd::span<double>(workspace_.multi_level_z->data(), total_level_scalars),
            block_dim_,
            rt);

        collect_multi_level_z<<<grid_num, 128, 0, rt.stream.get()>>>(
            ctd::span<const double>(workspace_.multi_level_z->data(), total_level_scalars),
            z,
            ctd::span<const int>(
                padded_topology_.real_to_padded->data(),
                padded_topology_.node_num),
            *coarse_space_.map,
            level_offsets,
            block_dim_,
            coarse_space_.level_num);
    }

    void invert_packed_matrices_for_test(ctd::span<double> mats, int block_dim, CudaRuntime rt)
    {
        int mat_dim = BANK_SIZE * block_dim;
        int mat_storage_size = mat_dim * (mat_dim + 1) / 2;
        int mat_num = mats.size() / mat_storage_size;
        invert_packed_matrices(mats.data(), mat_num, block_dim, rt);
    }

    void apply_packed_matrices_for_test(
        ctd::span<const double> mats,
        ctd::span<const double> x,
        ctd::span<double> y,
        int block_dim,
        CudaRuntime rt)
    {
        CoarseMatrices coarse_mats;
        coarse_mats.data = cu::make_buffer<double>(rt.stream, rt.mr, mats.size(), cu::no_init);
        cu::copy_bytes(rt.stream, mats, *coarse_mats.data);
        coarse_mats.total_matrix_num = 0;

        int mat_dim = BANK_SIZE * block_dim;
        int mat_storage_size = mat_dim * (mat_dim + 1) / 2;
        if (mat_storage_size > 0)
        {
            coarse_mats.total_matrix_num = mats.size() / mat_storage_size;
        }

        apply_packed_matrices(coarse_mats, x, y, block_dim, rt);
    }
} // namespace polysolve::linear::mas
