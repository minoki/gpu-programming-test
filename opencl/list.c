#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

char *getPlatformInfoAsString(cl_platform_id platform, cl_platform_info paramName)
{
    size_t paramSize;
    cl_int err = clGetPlatformInfo(platform, paramName, 0, NULL, &paramSize);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformInfo: %d\n", (int)err);
        exit(1);
    }
    char *data = malloc(paramSize);
    if (data == NULL) {
        fputs("malloc\n", stderr);
        exit(1);
    }
    err = clGetPlatformInfo(platform, paramName, paramSize, data, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformInfo: %d\n", (int)err);
        exit(1);
    }
    return data;
}

char *getDeviceInfoAsString(cl_device_id device, cl_device_info paramName)
{
    size_t paramSize;
    cl_int err = clGetDeviceInfo(device, paramName, 0, NULL, &paramSize);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetDeviceInfo: %d\n", (int)err);
        exit(1);
    }
    char *data = malloc(paramSize);
    if (data == NULL) {
        fputs("malloc\n", stderr);
        exit(1);
    }
    err = clGetDeviceInfo(device, paramName, paramSize, data, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetDeviceInfo: %d\n", (int)err);
        exit(1);
    }
    return data;
}

int main(void)
{
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
    for (cl_uint i = 0; i < numPlatforms; ++i) {
        printf("Platform #%u:\n", (unsigned int)i);
        char *platformProfile = getPlatformInfoAsString(platforms[i], CL_PLATFORM_PROFILE);
        printf("  Profile: %s\n", platformProfile);
        free(platformProfile);
        char *platformVersion = getPlatformInfoAsString(platforms[i], CL_PLATFORM_VERSION);
        printf("  Version: %s\n", platformVersion);
        free(platformVersion);
        char *platformName = getPlatformInfoAsString(platforms[i], CL_PLATFORM_NAME);
        printf("  Name: %s\n", platformName);
        free(platformName);
        char *platformVendor = getPlatformInfoAsString(platforms[i], CL_PLATFORM_VENDOR);
        printf("  Vendor: %s\n", platformVendor);
        free(platformVendor);
        char *platformExtensions = getPlatformInfoAsString(platforms[i], CL_PLATFORM_EXTENSIONS);
        printf("  Extensions: %s\n", platformExtensions);
        free(platformExtensions);
        cl_uint numDevices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clGetDeviceIDs: %d\n", (int)err);
            return 1;
        }
        cl_device_id *devices = malloc(sizeof(cl_device_id) * numDevices);
        if (devices == NULL) {
            fputs("malloc\n", stderr);
            return 1;
        }
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "clGetDeviceIDs: %d\n", (int)err);
            return 1;
        }
        for (cl_uint j = 0; j < numDevices; ++j) {
            cl_device_type deviceType;
            err = clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "clGetDeviceInfo: %d\n", (int)err);
                return 1;
            }
            printf("  Device %u (%s):\n", (unsigned int)j, deviceType == CL_DEVICE_TYPE_CPU ? "CPU" : deviceType == CL_DEVICE_TYPE_GPU ? "GPU" : deviceType == CL_DEVICE_TYPE_ACCELERATOR ? "accelerator" : deviceType == CL_DEVICE_TYPE_DEFAULT ? "default" : deviceType == CL_DEVICE_TYPE_CUSTOM ? "custom" : "unknown");
            char *deviceName = getDeviceInfoAsString(devices[j], CL_DEVICE_NAME);
            printf("    Name: %s\n", deviceName);
            free(deviceName);
            char *deviceVendor = getDeviceInfoAsString(devices[j], CL_DEVICE_VENDOR);
            printf("    Vendor: %s\n", deviceVendor);
            free(deviceVendor);
            char *driverVersion = getDeviceInfoAsString(devices[j], CL_DRIVER_VERSION);
            printf("    Driver version: %s\n", driverVersion);
            free(driverVersion);
            char *deviceProfile = getDeviceInfoAsString(devices[j], CL_DEVICE_PROFILE);
            printf("    Profile: %s\n", deviceProfile);
            free(deviceProfile);
            char *deviceVersion = getDeviceInfoAsString(devices[j], CL_DEVICE_VERSION);
            printf("    Device version: %s\n", deviceVersion);
            free(deviceVersion);
            char *openCLCVersion = getDeviceInfoAsString(devices[j], CL_DEVICE_OPENCL_C_VERSION);
            printf("    OpenCL C Version: %s\n", openCLCVersion);
            free(openCLCVersion);
            char *deviceExtensions = getDeviceInfoAsString(devices[j], CL_DEVICE_EXTENSIONS);
            printf("    Extensions: %s\n", deviceExtensions);
            free(deviceExtensions);
            char *deviceBuiltinKernels = getDeviceInfoAsString(devices[j], CL_DEVICE_BUILT_IN_KERNELS);
            printf("    Built-in kernels: %s\n", deviceBuiltinKernels);
            free(deviceBuiltinKernels);
            cl_uint maxComputeUnits, preferredVectorWidthHalf, preferredVectorWidthFloat, preferredVectorWidthDouble;
            err = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "clGetDeviceInfo: %d\n", (int)err);
                return 1;
            }
            printf("    Max compute units: %u\n", (unsigned int)maxComputeUnits);
            err = clGetDeviceInfo(devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, sizeof(preferredVectorWidthHalf), &preferredVectorWidthHalf, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "clGetDeviceInfo: %d\n", (int)err);
                return 1;
            }
            printf("    Preferred vector width (half): %u\n", (unsigned int)preferredVectorWidthHalf);
            err = clGetDeviceInfo(devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(preferredVectorWidthFloat), &preferredVectorWidthFloat, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "clGetDeviceInfo: %d\n", (int)err);
                return 1;
            }
            printf("    Preferred vector width (float): %u\n", (unsigned int)preferredVectorWidthFloat);
            err = clGetDeviceInfo(devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(preferredVectorWidthDouble), &preferredVectorWidthDouble, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "clGetDeviceInfo: %d\n", (int)err);
                return 1;
            }
            printf("    Preferred vector width (double): %u\n", (unsigned int)preferredVectorWidthDouble);
        }
        free(devices);
    }
    free(platforms);
}
