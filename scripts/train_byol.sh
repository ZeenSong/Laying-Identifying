#!/bin/bash
#SBATCH -J Train_BYOL
#SBATCH -p normal
#SBATCH -N 16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4
#SBATCH --mem=100G
#SBATCH -o Train_BYOL_8.o
#SBATCH -e Train_BYOL_8.e

hostfile=./$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}
num_node=$(cat $hostfile|sort|uniq |wc -l)

num_DCU=$(($num_node*4))
nodename=$(cat $hostfile |sed -n "1p")
dist_url=`echo $nodename | awk '{print $1}'`

rm `pwd`/hostfile-byol -f
cat $hostfile|sort|uniq >`pwd`/tmp

for i in `cat ./tmp`
do
    echo ${i} slots=4 >> `pwd`/hostfile-byol
done

module load /public/software/modules/apps/PyTorch/1.10.1a0/pytorch_1.10.1-rocm_4.0.1

echo "Start time: `date`" #显示开始时间
echo "SLURM_JOB_ID: $SLURM_JOB_ID" #显示作业号
echo "SLURM_NNODES: $SLURM_NNODES" #显示节点数
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE" #显示每节点任务数
echo "SLURM_NTASKS: $SLURM_NTASKS" #显示总任务数
echo "nodename: $nodename" #显示节点名
echo "dist_url: $dist_url" #显示主服务器
echo "num_DCU: $num_DCU" #显示总DCU数量

mpirun -np $num_DCU --allow-run-as-root -hostfile `pwd`/hostfile-byol ./train_single_byol.sh $dist_url