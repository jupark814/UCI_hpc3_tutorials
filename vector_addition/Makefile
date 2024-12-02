OBJ=vec_add

clean: 
	rm -f $(OBJ).o sumC sumC.o

cleanall: clean
	rm -f $(OBJ) *.out *.err /myoutput/*.raw h_A_B_C.txt

run: 
	sbatch job.slurm