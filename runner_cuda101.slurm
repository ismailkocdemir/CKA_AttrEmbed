#!/bin/bash

#SBATCH --nodes=1
#SBATCH -J abs2
#SBATCH --workdir=/home/ihakki/abstractness/	         
#SBATCH --gres=gpu:gtx1080:1                           
#SBATCH --output=/home/ihakki/abstractness/slurm_out/resnet18Vanilla_cifar100-slurm-%j.out	              
#SBATCH --error=/home/ihakki/abstractness/slurm_out/resnet18Vanilla_cifar100-slurm-%j.err	              
#SBATCH --time=2-00:00:00				                                

########### Cuda paths, you may not need that #############
export PATH=/home/ihakki/cuda-10.1/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/ihakki/cuda-10.1/lib64
#########################################################

source /home/ihakki/anaconda3/etc/profile.d/conda.sh
conda activate cuda101
bash train.sh 0 vico_select 3 

echo "JOB DONE!!!"
