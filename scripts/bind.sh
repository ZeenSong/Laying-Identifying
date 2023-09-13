#!/bin/bash
APPCMD="$*"
lrank=$(expr $OMPI_COMM_WORLD_LOCAL_RANK % 4)
case ${lrank} in
[0])
 export HIP_VISIBLE_DEVICES=0,1,2,3
 export UCX_NET_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3:1
 export UCX_IB_PCI_BW=mlx5_0,mlx5_1,mlx5_2,mlx5_3:50Gbs
 export NCCL_IB_DISABLE=1
 export NCCL_P2P_DISABLE=1
 numactl --cpunodebind=1 --membind=1 ${APPCMD}
 ;;
[1])
 export HIP_VISIBLE_DEVICES=1
 export UCX_NET_DEVICES=mlx5_1:1
 export UCX_IB_PCI_BW=mlx5_1:50Gbs
 numactl --cpunodebind=1 --membind=1 ${APPCMD}
 ;;
[2])
 export HIP_VISIBLE_DEVICES=2
 export UCX_NET_DEVICES=mlx5_2:1
 export UCX_IB_PCI_BW=mlx5_2:50Gbs
 numactl --cpunodebind=2 --membind=2 ${APPCMD}
 ;;
[3])
 export HIP_VISIBLE_DEVICES=3
 export UCX_NET_DEVICES=mlx5_3:1
 export UCX_IB_PCI_BW=mlx5_3:50Gbs
 numactl --cpunodebind=3 --membind=3 ${APPCMD}
 ;;
esac