#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Vector Addition

// Device Code
__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<len) out[i] = in1[i] + in2[i];
}


// Host Code
float *readCMD(const char *fileName, int *length) {
    FILE *file = fopen(fileName, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    // Read the length from the first line
    if (fscanf(file, "%d", length) != 1) {
        fprintf(stderr, "Error: Could not read length from file %s\n", fileName);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    // printf("Reading file: %s, Length: %d\n", fileName, *length); // Debug print

    // Allocate memory for the array
    float *data = (float *)malloc(*length * sizeof(float));
    if (!data) {
        fprintf(stderr, "Error: Could not allocate memory for input data\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Read the array data
    for (int i = 0; i < *length; i++) {
        if (fscanf(file, "%f", &data[i]) != 1) {
            fprintf(stderr, "Error: Could not read data from file %s\n", fileName);
            free(data);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
    /*
    // Print the first few elements for debugging
    printf("Data from %s:\n", fileName);
    for (int i = 0; i < (*length < 5 ? *length : 5); i++) { // Print up to 5 elements
        printf("Element %d: %f\n", i, data[i]);
    }
    */
    return data;
}


void saveOutput(const char *fileName, float *data, int length) {
    FILE *file = fopen(fileName, "w");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", fileName);
        exit(EXIT_FAILURE);
    }

    // Write the length of the array (optional, for debugging purposes)
    fprintf(file, "%d\n", length);

    // Write the data
    for (int i = 0; i < length; i++) {
        fprintf(file, "%f\n", data[i]);
    }

    fclose(file);
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

  // printf("Input file 1: %s\n", file1); // Debug print
  // printf("Input file 2: %s\n", file2); // Debug print

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

  // Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(inputLength/256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);

  // Launch the GPU Kernel here to perform CUDA computation
  vecAdd<<<DimGrid,DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize();

  // Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);

  // Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  // printf("It is successful until here\n");
  saveOutput("/tmp/myoutput.raw", hostOutput, inputLength);

  // Save the output to the output file
  FILE *output = fopen(outputFile, "w");
  if (!output) {
      fprintf(stderr, "Error: Could not open output file %s for writing\n", outputFile);
      free(hostInput1);
      free(hostInput2);
      free(hostOutput);
      return EXIT_FAILURE;
  }

  // Write the length as the first line
  fprintf(output, "%d\n", inputLength);

  // Write each element of the output array on a new line
  for (int i = 0; i < inputLength; i++) {
    if (hostOutput[i] == (int)hostOutput[i]) {
        // Value is a whole number, print as integer
        fprintf(output, "%d.", (int)hostOutput[i]);
    } else {
        // Value is not a whole number, print as float with one decimal place
        fprintf(output, "%.1f", hostOutput[i]);
    }
    if ( i+1 != inputLength ) {
      fprintf(output, "\n");
    }
  }

  fclose(output);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}