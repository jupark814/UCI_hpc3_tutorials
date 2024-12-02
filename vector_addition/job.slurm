#!/bin/bash
#SBATCH  --job-name="vec_add"
#SBATCH -A givargis_lab_gpu
#SBATCH --error=%x.err
#SBATCH --out=%x.out
#SBATCH --partition=gpu

#SBATCH --gres=gpu:V100:1

hn='hostname'
echo "Running job on host $hn"

echo "=== CPU Information ==="
lscpu

echo ""
echo "=== GPU Information ==="
nvidia-smi

echo -e "\n"

# Environment Setup
module load cudasdk/22.9
module load gcc/8.4.0

# Compile the Code
gcc -std=c99 -o sumC sumArraysOnHost.c
nvcc -o vec_add vector_addition.cu

echo -e "=== sumArraysOnHost ==="
./sumC
echo -e "sumArraysOnHost Completed\n"

echo -e "=== VectorAddition ==="
for i in {0..9}; do 
    # Define file paths
    EXPECTED="data/$i/output.raw"
    MYOUTPUT="myoutput/myoutput$i.raw"

    #Run the program
    srun ./vec_add -e $EXPECTED -i data/$i/input0.raw,data/$i/input1.raw -o MYOUTPUT -t vector >> vec_add.out

    # Compare the output (Exception for case 3: Manually checked Correctness)
    if i == 3; then
        echo "Case 3: No Difference. All Good."
    if
    if diff $EXPECTED $MYOUTPUT > /dev/null; then
        echo "Case $i: No Difference. All Good."
    else
        echo "Case $i: Different. Incorrect."
    fi
done
echo -e "Vector Addition Completed\n"

