
template <int N, int BLOCK>
__global__ void symv_algo6_upper_shared(const double *__restrict__ A_upper,
                                        const double *__restrict__ x,
                                        double *__restrict__ y,
                                        int num_mats)
{
    int mat = int(blockIdx.x);
    if (mat >= num_mats)
        return;

    int row = int(threadIdx.x);

    constexpr int L = N * (N + 1) / 2;
    const double *Amat = A_upper + size_t(mat) * L;

    __shared__ double sx[N];
    __shared__ double sA[L];

    for (int i = int(threadIdx.x); i < N; i += BLOCK)
        sx[i] = x[size_t(mat) * N + i];
    for (int k = int(threadIdx.x); k < L; k += BLOCK)
        sA[k] = Amat[k];

    __syncthreads();

    if (row >= N)
        return;

    double sum = 0.0;
#pragma unroll
    for (int col = 0; col < N; ++col)
    {
        int r = row <= col ? row : col;
        int c = row <= col ? col : row;
        sum += sA[upper_index<N>(r, c)] * sx[col];
    }

    y[size_t(mat) * N + row] = sum;
}
