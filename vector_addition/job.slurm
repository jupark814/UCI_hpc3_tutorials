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
module load cudasdk/22.9

# nvcc -Xcompiler -std=c99 sumArraysOnHost.c -o sumC

# echo -e "=== sumArraysOnHost ==="
# ./sumC

nvcc -o vec_add vector_addition.cu
echo -e "=== VectorAddition ==="
srun ./vec_add -e data/0/output.raw -i data/0/input0.raw,data/0/input1.raw -o myoutput/myoutput0.raw -t vector >> vec_add.out
srun ./vec_add -e data/1/output.raw -i data/1/input0.raw,data/1/input1.raw -o myoutput/myoutput1.raw -t vector >> vec_add.out
srun ./vec_add -e data/2/output.raw -i data/2/input0.raw,data/2/input1.raw -o myoutput/myoutput2.raw -t vector >> vec_add.out
srun ./vec_add -e data/3/output.raw -i data/3/input0.raw,data/3/input1.raw -o myoutput/myoutput3.raw -t vector >> vec_add.out
srun ./vec_add -e data/4/output.raw -i data/4/input0.raw,data/4/input1.raw -o myoutput/myoutput4.raw -t vector >> vec_add.out
srun ./vec_add -e data/5/output.raw -i data/5/input0.raw,data/5/input1.raw -o myoutput/myoutput5.raw -t vector >> vec_add.out
srun ./vec_add -e data/6/output.raw -i data/6/input0.raw,data/6/input1.raw -o myoutput/myoutput6.raw -t vector >> vec_add.out
srun ./vec_add -e data/7/output.raw -i data/7/input0.raw,data/7/input1.raw -o myoutput/myoutput7.raw -t vector >> vec_add.out
srun ./vec_add -e data/8/output.raw -i data/8/input0.raw,data/8/input1.raw -o myoutput/myoutput8.raw -t vector >> vec_add.out
srun ./vec_add -e data/9/output.raw -i data/9/input0.raw,data/9/input1.raw -o myoutput/myoutput9.raw -t vector >> vec_add.out