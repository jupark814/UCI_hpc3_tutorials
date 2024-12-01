#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int idx=0; idx<N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

void initialData(float *ip, int size) {
    // generate different seed for random number
    time_t t;
    srand((unsigned int) time(&t));
    for (int i=0; i<size; i++) {
        ip[i] = (float) ( rand() & 0xFF ) / 10.0f;
    }
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

int main (int argc, char **argv) {
    int nElem = 10;
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    sumArraysOnHost(h_A, h_B, h_C, nElem);

    saveArrayToFile("h_A_B_C.txt", h_A, nElem);
    saveArrayToFile("h_A_B_C.txt", h_B, nElem);
    saveArrayToFile("h_A_B_C.txt", h_C, nElem);

    free(h_A);
    free(h_B);
    free(h_C);

    return(0);
}
