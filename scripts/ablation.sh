#!/bin/bash
#SBATCH -J Ablation_BYOL #作业名称
#SBATCH -p debug #使用debug分区
#SBATCH -N 1 #使用一个节点
#SBATCH --cpus-per-task=32 #每个节点使用32个CPU
#SBATCH --gres=dcu:4 #使用4张加速卡
#SBATCH --mem=50G #使用10G内存
#SBATCH -o ablation_1.o 
#SBATCH -e ablation_1.e 

echo "Start time: `date`" #显示开始时间
echo "SLURM_JOB_ID: $SLURM_JOB_ID" #显示作业号
echo "SLURM_NNODES: $SLURM_NNODES" #显示节点数
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE" #显示每节点任务数
echo "SLURM_NTASKS: $SLURM_NTASKS" #显示总任务数
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION" #显示作业分区

module load /public/software/modules/apps/PyTorch/1.10.1a0/pytorch_1.10.1-rocm_4.0.1 #加载环境
python -m train --dataset cifar10 --method byol --lr 3e-3 --emb 64 --eval_every 20 --num_workers 8 --IL 1 --log byol_2.log --bs 256 --epoch 100 --yeta 15 --pretrained-path ./resnet18-5c106cde.pth #运行任务

echo "End time: `date`" #显示结束时间