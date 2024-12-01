#!/bin/bash
#SBATCH --job-name="helloworld"
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

nvcc -arch sm_70 helloworld.cu -o helloworld

echo -e "=== helloworld ==="
./helloworld
