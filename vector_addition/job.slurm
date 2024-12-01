#!/bin/bash
#SBATCH  --job-name="vector_addition"
#SBATCH -A givargis_lab_gpu
#SBATCH --error=%x.error
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

nvcc -Xcompiler -std=c99 sumArraysOnHost.c -o sumC

echo -e "=== sumArraysOnHost ==="
./sumC