#include <stdio.h>
#include <stdlib.h>

// Vector Addition

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<len) out[i] = in1[i] + in2[i];
}

// Function to read data from binary file
float* readBinaryFile(const char *filename, int *length) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", filename);
        exit(1);
    }

    // Determine file size
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    rewind(file);

    // Calculate the number of elements (assuming float data)
    *length = fileSize / sizeof(float);

    // Allocate memory and read data
    float *data = (float *)malloc(fileSize);
    if (data == NULL) {
        printf("Error: Unable to allocate memory for file %s\n", filename);
        fclose(file);
        exit(1);
    }
    fread(data, sizeof(float), *length, file);
    fclose(file);

    return data;
}

void saveArrayToFile(const char *filename, float *array, int size) {
	FILE *file = fopen(filename, "a");
	if (file == NULL) {
		printf("Error opening file %s for writing.\n", filename);
		exit(1);
	}
	fprintf(file, "\n === ARRAY === \n");
	for (int i = 0; i < size; i++) {
		fprintf(file, "%f\n", array[i]); // Write each element to a new line
	}
	fprintf(file, "\n");
	fclose(file);
}

int main(int argc, char **argv) {
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  //@@ Importing data and creating memory on host
  hostInput1 = readBinaryFile("/pub/dpark15/ece408/UIUC_ECE408/lab1/data/0/input0.raw", &inputLength);
  hostInput2 = readBinaryFile("/pub/dpark15/ece408/UIUC_ECE408/lab1/data/0/input0.raw", &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  
  int size = inputLength*sizeof(float);

  //@@ Allocate GPU memory here
  cudaMalloc((void**) &deviceInput1, size);
  cudaMalloc((void**) &deviceInput2, size);
  cudaMalloc((void**) &deviceOutput, size);

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(inputLength/256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);

  //@@ Launch the GPU Kernel here to perform CUDA computation
  vecAdd<<<DimGrid,DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
  saveArrayToFile("d_C.txt", h_B, nElem);

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}