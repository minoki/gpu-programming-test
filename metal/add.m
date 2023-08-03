#include <stdio.h>
#import <Metal/Metal.h>

static const char kernel_src[] =
"kernel void add(device const float *a, device const float *b, device float *result, uint index [[thread_position_in_grid]])\n"
"{\n"
"    result[index] = a[index] + b[index];\n"
"}\n";

int main(void)
{
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            fputs("Failed to create system default device.\n", stderr);
            return 1;
        }
        printf("device name: %s\n", [[device name] UTF8String]);
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (commandQueue == nil) {
            fputs("Failed to find the command queue.\n", stderr);
            return 1;
        }

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kernel_src] options:[MTLCompileOptions new] error:&error];
        if (library == nil) {
            NSLog(@"Failed to compile source: %@", error);
            return 1;
        }

        id<MTLFunction> addFunction = [library newFunctionWithName:@"add"];
        if (addFunction == nil) {
            fputs("Failed to find add function\n", stderr);
            return 1;
        }

        id<MTLComputePipelineState> addFunctionPSO = [device newComputePipelineStateWithFunction:addFunction error:&error];
        if (addFunctionPSO == nil) {
            NSLog(@"Failed to create pipeline state object: %@", error);
            return 1;
        }

        float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float b[] = {2.0f, 7.0f, -4.0f, 7.2f, 9.5f};
        const size_t n = 5;

        id<MTLBuffer> bufA = [device newBufferWithBytes:a length:sizeof(a) options:MTLResourceStorageModeManaged];
        id<MTLBuffer> bufB = [device newBufferWithBytes:b length:sizeof(b) options:MTLResourceStorageModeManaged];
        id<MTLBuffer> bufC = [device newBufferWithLength:sizeof(float) * n options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:addFunctionPSO];
        [computeEncoder setBuffer:bufA offset:0 atIndex:0];
        [computeEncoder setBuffer:bufB offset:0 atIndex:1];
        [computeEncoder setBuffer:bufC offset:0 atIndex:2];

        {
            MTLSize gridSize = MTLSizeMake(n, 1, 1);

            NSUInteger threadGroupSize = addFunctionPSO.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > n) {
                threadGroupSize = n;
            }
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        }

        [computeEncoder endEncoding];

        [commandBuffer commit];

        [commandBuffer waitUntilCompleted];

        const float *c = [bufC contents];
        for (size_t i = 0; i < n; ++i) {
            printf("c[%zu] = %g\n", i, c[i]);
        }
    }
}
