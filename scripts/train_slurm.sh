#!/bin/bash
#SBATCH -J ImageNet_eval
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=dcu:4
#SBATCH --mem=100G
#SBATCH --exclusive
#SBATCH -o ImageNet_test.o
#SBATCH -e ImageNet_test.e

echo "Start time: `date`" #显示开始时间
echo "SLURM_JOB_ID: $SLURM_JOB_ID" #显示作业号
echo "SLURM_NNODES: $SLURM_NNODES" #显示节点数
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE" #显示每节点任务数
echo "SLURM_NTASKS: $SLURM_NTASKS" #显示总任务数
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION" #显示作业分区

module load /public/software/modules/apps/PyTorch/1.10.1a0/pytorch_1.10.1-rocm_4.0.1
python -m test_multi --dataset imagenet --method byol --fname byol_imagenet.pt --imagenet_path /public/home/iscasai/songzeen/data/ImageNet/ --IL 1 --emb 128 --arch resnet50 --log test_imagenet.log --pretrained-path ./resnet50-19c8e357.pth --num_workers 4

echo "End time: `date`" #显示结束时间
