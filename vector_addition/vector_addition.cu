#include <stdio.h>
#include <stdlib.h>

// Vector Addition

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<len) out[i] = in1[i] + in2[i];
}

float *readCMD(const char *fileName, int *length) {
    FILE *file = fopen(fileName, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    // Read the length of the array from the file (assuming length is stored in the first 4 bytes)
    fread(length, sizeof(int), 1, file);
    printf("Reading file: %s, Length: %d\n", fileName, *length); // Debug print
    
    // Allocate memory for the array
    float *data = (float *)malloc(*length * sizeof(float));
    if (!data) {
        fprintf(stderr, "Error: Could not allocate memory for input data\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Read the array data
    fread(data, sizeof(float), *length, file);
    fclose(file);
    return data;
}

int main(int argc, char **argv) {
  if (argc < 4) {
      fprintf(stderr, "Usage: %s -i input0.raw,input1.raw -o output.raw -t vector\n", argv[0]);
      return EXIT_FAILURE;
  }

  char *inputFiles = NULL;
  char *outputFile = NULL;

  // Parse command-line arguments
  for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
          inputFiles = argv[++i];
      } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
          outputFile = argv[++i];
      }
  }

  if (!inputFiles || !outputFile) {
      fprintf(stderr, "Error: Missing required arguments\n");
      return EXIT_FAILURE;
  }

  // Split input files
  char *file1 = strtok(inputFiles, ",");
  char *file2 = strtok(NULL, ",");
  if (!file1 || !file2) {
      fprintf(stderr, "Error: Invalid input file format\n");
      return EXIT_FAILURE;
  }

  int inputLength;
  float *hostInput1 = readCMD(file1, &inputLength);
  float *hostInput2 = readCMD(file2, &inputLength);
  float *hostOutput = (float *)malloc(inputLength * sizeof(float));
  if (!hostOutput) {
      fprintf(stderr, "Error: Could not allocate memory for output\n");
      free(hostInput1);
      free(hostInput2);
      return EXIT_FAILURE;
  }

  // int inputLength;
  // float *hostInput1;
  // float *hostInput2;
  // float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  //@@ Importing data and creating memory on host
  // hostInput1 = readCMD(file1, &inputLength);
  // hostInput2 = readCMD(file2, &inputLength);
  // hostOutput = (float *)malloc(inputLength * sizeof(float));
  
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

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  printf("It is successful until here\n");

  //@@ Save the output to the output file
  FILE *output = fopen(outputFile, "wb");
  if (!output) {
      fprintf(stderr, "Error: Could not open output file %s\n", outputFile);
      free(hostInput1);
      free(hostInput2);
      free(hostOutput);
      return EXIT_FAILURE;
  }

  // Write the array length and data
  fwrite(&inputLength, sizeof(int), 1, output);
  fwrite(hostOutput, sizeof(float), inputLength, output);
  fclose(output);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}