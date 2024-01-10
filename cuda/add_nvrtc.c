#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>

static const char kernel_src[] =
"extern \"C\" __global__ void add(int n, const float *a, const float *b, float *c)\n"
"{\n"
"    int i = blockDim.x * blockIdx.x + threadIdx.x;\n"
"    if (i < n) {\n"
"        c[i] = a[i] + b[i];\n"
"    }\n"
"}\n";

static void compile(CUdevice cuDevice, const char *src, char **result, size_t *resultSize)
{
    cudaError_t cuErr = cudaSuccess;
    nvrtcResult err = NVRTC_SUCCESS;
    nvrtcProgram prog;
    err = nvrtcCreateProgram(&prog, src, "add.cu", /* numHeaders */ 0, /* headers*/ NULL, /* includeNames */ NULL);
    if (err != NVRTC_SUCCESS) {
        fprintf(stderr, "Failed to create program (%s)\n", nvrtcGetErrorString(err));
        exit(1);
    }

    int major = 0, minor = 0;
    cuErr = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
    if (cuErr != cudaSuccess) {
        fprintf(stderr, "Failed to get major version (%s)\n", cudaGetErrorString(cuErr));
        exit(1);
    }
    cuErr = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);
    if (cuErr != cudaSuccess) {
        fprintf(stderr, "Failed to get minor version (%s)\n", cudaGetErrorString(cuErr));
        exit(1);
    }

    char compileOptionBuf[sizeof("--gpu-architecture=sm_XXXXXXXXXX")];
    snprintf(compileOptionBuf, sizeof(compileOptionBuf), "--gpu-architecture=sm_%d%d", major, minor);
    const char *compileOptions[] = {compileOptionBuf};
    err = nvrtcCompileProgram(prog, 1, compileOptions);
    if (err != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcResult err2 = nvrtcGetProgramLogSize(prog, &logSize);
        if (err2 != NVRTC_SUCCESS) {
            fprintf(stderr, "Failed to get program log size (%s)\n", nvrtcGetErrorString(err2));
            exit(1);
        }
        char *log = malloc(logSize);
        err2 = nvrtcGetProgramLog(prog, log);
        if (err2 != NVRTC_SUCCESS) {
            fprintf(stderr, "Failed to get program log (%s)\n", nvrtcGetErrorString(err2));
            exit(1);
        }
        fprintf(stderr, "Failed to compile (%s): %s", nvrtcGetErrorString(err), log);
        free(log);
        exit(1);
    }
    size_t ptxSize;
    err = nvrtcGetPTXSize(prog, &ptxSize);
    if (err != NVRTC_SUCCESS) {
        fprintf(stderr, "Failed to get PTX size (%s)\n", nvrtcGetErrorString(err));
        exit(1);
    }
    char *ptx = malloc(ptxSize);
    err = nvrtcGetPTX(prog, ptx);
    if (err != NVRTC_SUCCESS) {
        fprintf(stderr, "Failed to get PTX (%s)\n", nvrtcGetErrorString(err));
        exit(1);
    }
    *result = ptx;
    *resultSize = ptxSize;

    nvrtcDestroyProgram(&prog);
}

int main(int argc, char *argv[])
{
    cudaError_t err = cudaSuccess;

    err = cuInit(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to initialize (%s)\n", cudaGetErrorString(err));
        return 1;
    }

    int deviceCount = 0;
    err = cuDeviceGetCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count (%s)\n", cudaGetErrorString(err));
        return 1;
    }
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA device found\n");
        exit(1);
    }

    CUdevice cuDevice;
    err = cuDeviceGet(&cuDevice, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device (%s)\n", cudaGetErrorString(err));
        return 1;
    }

    CUcontext cuContext;
    err = cuCtxCreate(&cuContext, 0, cuDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to create context (%s)\n", cudaGetErrorString(err));
        return 1;
    }

    char *ptx = NULL;
    size_t ptxSize = 0;
    compile(cuDevice, kernel_src, &ptx, &ptxSize);

    CUmodule module;
    err = cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to load module (%s)\n", cudaGetErrorString(err));
        return 1;
    }

    free(ptx);

    CUfunction kernel;
    err = cuModuleGetFunction(&kernel, module, "add");
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get function (%s)\n", cudaGetErrorString(err));
        return 1;
    }

    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float b[] = {2.0f, 7.0f, -4.0f, 7.2f, 9.5f};
    const size_t n = 5;

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

    // add<<<blocksPerGrid, threadsPerBlock>>>(n, device_a, device_b, device_c);
    void *args[] = {&n, &device_a, &device_b, &device_c};
    err = cuLaunchKernel(kernel, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, /* sharedMemBytes */ 0, /* hStream */ NULL, args, 0);
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
