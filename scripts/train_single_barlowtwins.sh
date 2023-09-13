#!/bin/bash
export MIOPEN_USER_DB_PATH=/tmp/pytorch-miopen-2.8
export HSA_USERPTR_FOR_PAGED_MEM=0
export NCCL_IB_DISABLE=1


lrank=$OMPI_COMM_WORLD_LOCAL_RANK
comm_rank=$OMPI_COMM_WORLD_RANK
comm_size=$OMPI_COMM_WORLD_SIZE

APP="python train.py --bs 4096 --dataset imagenet --head_size 1024 --epoch 100 --lr 1e-3 --emb 1024 --crop_s0 0.08 --cj0 0.8 --cj1 0.8 --cj2 0.8 --cj3 0.2 --gs_p 0.2  --method barlowtwins --num_workers 8 --IL 1 --arch resnet50 --mu 2 --log barlow_1.log --pretrained-path ./resnet18-5c106cde.pth --imagenet_path ./data/ImageNet100/ --dist-url tcp://${1}:25837 --world-size=${comm_size} --rank=${comm_rank}"
case ${lrank} in
[0])
  export HIP_VISIBLE_DEVICES=0
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_PCI_BW=mlx5_0:50Gbs
  numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[1])
  export HIP_VISIBLE_DEVICES=1
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
  numactl --cpunodebind=1 --membind=1 ${APP}
  ;;
[2])
  export HIP_VISIBLE_DEVICES=2
  export UCX_NET_DEVICES=mlx5_2:1
  export UCX_IB_PCI_BW=mlx5_2:50Gbs
  numactl --cpunodebind=2 --membind=2 ${APP}
  ;;
[3])
  export HIP_VISIBLE_DEVICES=3
  export UCX_NET_DEVICES=mlx5_3:1
  export UCX_IB_PCI_BW=mlx5_3:50Gbs
  numactl --cpunodebind=3 --membind=3 ${APP}
  ;;
esac
