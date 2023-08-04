#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

static const char kernel_src[] =
"__kernel void add(int n, __global const float *a, __global const float *b, __global float *c)\n"
"{\n"
"    int i = get_global_id(0);\n"
"    if (i < n) {\n"
"        c[i] = a[i] + b[i];\n"
"    }\n"
"}\n";

static const char *errorToString(cl_int e)
{
    switch (e) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    default: return "<unknown>";
    }
}

int main(int argc, char *argv[])
{
    int platformIndex = 0, deviceIndex = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--platform") == 0 && i + 1 < argc) {
            platformIndex = atoi(argv[i + 1]);
            if (platformIndex < 0) {
                platformIndex = 0;
            }
            ++i;
        }
        if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            deviceIndex = atoi(argv[i + 1]);
            if (deviceIndex < 0) {
                deviceIndex = 0;
            }
            ++i;
        }
    }
    cl_uint numPlatforms;
    cl_int err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformIDs: %d\n", (int)err);
        return 1;
    }
    cl_platform_id *platforms = malloc(sizeof(cl_platform_id) * numPlatforms);
    if (platforms == NULL) {
        fputs("malloc\n", stderr);
        return 1;
    }
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformIDs: %d\n", (int)err);
        return 1;
    }
    if (platformIndex >= numPlatforms) {
        fputs("Platform index too large.\n", stderr);
        return 1;
    }
    cl_platform_id platform = platforms[platformIndex];
    free(platforms);
    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetDeviceIDs: %d\n", (int)err);
        return 1;
    }
    cl_device_id *devices = malloc(sizeof(cl_device_id) * numDevices);
    if (devices == NULL) {
        fputs("malloc\n", stderr);
        return 1;
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetDeviceIDs: %d\n", (int)err);
        return 1;
    }
    if (deviceIndex >= numDevices) {
        fputs("Device index too large.\n", stderr);
        return 1;
    }
    cl_device_id device = devices[deviceIndex];

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (context == NULL) {
        fprintf(stderr, "clCreateContext: %d\n", (int)err);
        return 1;
    }

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (queue == NULL) {
        fprintf(stderr, "clCreateCommandQueue: %d\n", (int)err);
        return 1;
    }

    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float b[] = {2.0f, 7.0f, -4.0f, 7.2f, 9.5f};
    const size_t n = 5;

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(a), NULL, &err);
    if (bufA == NULL) {
        fprintf(stderr, "clCreateBuffer: %d\n", (int)err);
        return 1;
    }
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(b), NULL, &err);
    if (bufB == NULL) {
        fprintf(stderr, "clCreateBuffer: %d\n", (int)err);
        return 1;
    }
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n, NULL, &err);
    if (bufC == NULL) {
        fprintf(stderr, "clCreateBuffer: %d\n", (int)err);
        return 1;
    }

    const char *kernel_src_ = kernel_src;
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_src_, NULL, &err);
    if (program == NULL) {
        fprintf(stderr, "clCreateProgramWithSource: %d\n", (int)err);
        return 1;
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clBuildProgram: %s (%d)\n", errorToString(err), (int)err);
        size_t paramSize;
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &paramSize);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clGetProgramBuildInfo: %s (%d)\n", errorToString(err), (int)err);
            return 1;
        }
        char *buildLog = malloc(paramSize);
        if (buildLog == NULL) {
            fputs("malloc", stderr);
            return 1;
        }
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, paramSize, buildLog, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clGetProgramBuildInfo: %s (%d)\n", errorToString(err), (int)err);
            return 1;
        }
        fprintf(stderr, "Build log:\n%s\n", buildLog);
        free(buildLog);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "add", &err);
    if (kernel == NULL) {
        fprintf(stderr, "clCreateKernel: %d\n", (int)err);
        return 1;
    }

    {
        cl_int n_ = n;
        err = clSetKernelArg(kernel, 0, sizeof(cl_int), &n_);
    }
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", (int)err);
        return 1;
    }
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufA);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", (int)err);
        return 1;
    }
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufB);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", (int)err);
        return 1;
    }
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufC);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg: %d\n", (int)err);
        return 1;
    }

    err = clEnqueueWriteBuffer(queue, bufA, CL_FALSE, 0, sizeof(a), a, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueWriteBuffer: %d\n", (int)err);
        return 1;
    }
    err = clEnqueueWriteBuffer(queue, bufB, CL_FALSE, 0, sizeof(b), b, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueWriteBuffer: %d\n", (int)err);
        return 1;
    }

    size_t localWorkSize = 256;
    size_t globalWorkSize = n % localWorkSize == 0 ? n : n + localWorkSize - (n % localWorkSize);
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueNDRangeKernel: %s (%d)\n", errorToString(err), (int)err);
        return 1;
    }

    float c[5];
    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * n, c, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueReadBuffer: %d\n", (int)err);
        return 1;
    }

    for (size_t i = 0; i < n; ++i) {
        printf("c[%zu] = %g\n", i, c[i]);
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
}
