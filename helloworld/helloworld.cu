#include <stdio.h>

__global__ void helloFromGPU (void)
{
    printf("Hello World from GPU with blockidx.x: %d, threadidx.x: %d!\n", blockIdx.x, threadIdx.x);
}

int main(void)
{
    // hello from cpu
    printf("Hello World from CPU!\n");

    // hello from GPU
    helloFromGPU<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}
