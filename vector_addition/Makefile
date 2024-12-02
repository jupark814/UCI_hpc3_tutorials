OBJ=vec_add

all: 
	nvcc -o vec_add vector_addition.cu

clean: 
	rm -f $(OBJ).o

cleanall: clean
	rm -f $(OBJ) *.out *.err

run: 
	sbatch job.slurm