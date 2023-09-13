export MIOPEN_USER_DB_PATH=/tmp/pytorch-miopen-2.8
export HSA_USERPTR_FOR_PAGED_MEM=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1


lrank=$OMPI_COMM_WORLD_LOCAL_RANK
comm_rank=$OMPI_COMM_WORLD_RANK
comm_size=$OMPI_COMM_WORLD_SIZE

APP="python test_multi.py --dataset imagenet --method byol --fname byol_imagenet.pt --imagenet_path /public/home/iscasai/songzeen/data/ImageNet/ --IL 1 --emb 128 --arch resnet50 --log test_imagenet_byol.log --pretrained-path ./resnet50-19c8e357.pth --num_workers 4 --dist-url tcp://${1}:35246 --world-size=${comm_size} --rank=${comm_rank}"
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