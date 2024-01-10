#include <stdio.h>

__global__ void add(int n, const float *a, const float *b, float *c)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[])
{
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float b[] = {2.0f, 7.0f, -4.0f, 7.2f, 9.5f};
    const size_t n = 5;

    cudaError_t err = cudaSuccess;

    float *device_a = NULL;
    err = cudaMalloc((void **)&device_a, sizeof(a));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (%s)\n", cudaGetErrorString(err));
        return 1;
    }

    float *device_b = NULL;
    err = cudaMalloc((void **)&device_b, sizeof(b));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (%s)\n", cudaGetErrorString(err));
        return 1;
    }

    float *device_c = NULL;
    err = cudaMalloc((void **)&device_c, n * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (%s)\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(device_a, a, sizeof(a), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (%s)\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(device_b, b, sizeof(b), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from host to device (%s)\n", cudaGetErrorString(err));
        return 1;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    add<<<blocksPerGrid, threadsPerBlock>>>(n, device_a, device_b, device_c);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch add kernel (%s)\n", cudaGetErrorString(err));
        return 1;
    }

    float c[5];
    err = cudaMemcpy(c, device_c, sizeof(c), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from device to host (%s)\n", cudaGetErrorString(err));
        return 1;
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    for (size_t i = 0; i < n; ++i) {
        printf("c[%zu] = %g\n", i, c[i]);
    }
}
