
__global__ void batched_invert_upper_packed_96x96(double *d_matrices,
                                                  int mat_num)
{
    int mat_idx = blockIdx.x;
    if (mat_idx >= mat_num)
        return;

    // Each block processes exactly one 96x96 matrix (4656 elements)
    double *d_A = d_matrices + mat_idx * 4656;

    __shared__ double s_A[4656];
    int tx = threadIdx.x;

    // Collaborative load from global to shared memory
    for (int i = tx; i < 4656; i += 96)
    {
        s_A[i] = d_A[i];
    }
    __syncthreads();

    // Inline lambda for row-major upper packed indexing
    auto idx = [](int r, int c) { return r * 96 - (r * (r + 1)) / 2 + c; };

    // Phase 1: In-place Cholesky Factorization (A = U^T U)
    for (int i = 0; i < 96; i++)
    {
        if (tx == i)
        {
            s_A[idx(i, i)] = sqrt(s_A[idx(i, i)]);
        }
        __syncthreads();

        if (tx > i)
        {
            s_A[idx(i, tx)] /= s_A[idx(i, i)];
        }
        __syncthreads();

        if (tx > i)
        {
            double U_i_tx = s_A[idx(i, tx)];
            for (int c = tx; c < 96; c++)
            {
                s_A[idx(tx, c)] -= U_i_tx * s_A[idx(i, c)];
            }
        }
        __syncthreads();
    }

    // Phase 2: Invert U in-place (U^-1)
    // Processes column by column left-to-right.
    for (int c = 0; c < 96; c++)
    {
        double inv_Ucc = 1.0 / s_A[idx(c, c)];
        double new_val = 0.0;

        if (tx < c)
        {
            double sum = 0.0;
            for (int k = tx; k < c; k++)
            {
                sum += s_A[idx(tx, k)] * s_A[idx(k, c)];
            }
            new_val = -sum * inv_Ucc;
        }
        __syncthreads();

        if (tx < c)
        {
            s_A[idx(tx, c)] = new_val;
        }
        else if (tx == c)
        {
            s_A[idx(c, c)] = inv_Ucc;
        }
        __syncthreads();
    }

    // Phase 3: Matrix Multiplication A^-1 = U^-1 * U^-T
    // Processes column by column left-to-right.
    for (int c = 0; c < 96; c++)
    {
        double new_val = 0.0;

        if (tx <= c)
        {
            double sum = 0.0;
            for (int k = c; k < 96; k++)
            {
                sum += s_A[idx(tx, k)] * s_A[idx(c, k)];
            }
            new_val = sum;
        }
        __syncthreads();

        if (tx <= c)
        {
            s_A[idx(tx, c)] = new_val;
        }
        __syncthreads();
    }

    // Collaborative store from shared back to global memory
    for (int i = tx; i < 4656; i += 96)
    {
        d_A[i] = s_A[i];
    }
}
