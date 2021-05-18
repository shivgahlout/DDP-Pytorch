# DDP-Pytorch
Distributed computing in PyTorch with Distributed Data Parallel (DDP)

#### The detailed related blog can be found [here](https://shivgahlout.github.io/2021-05-18-distributed-computing/).

#### train_distributed.py is without the module `torch.distributed.launch`. On node 0, launch it as:

````python
python train_distributed.py --total_nodes 2 --gpus 2 --node_rank 0
````
On node 1, launch it as: 
````Python
python train_distributed.py --total_nodes 2 --gpus 2 --node_rank 1
````
and so on.
  
#### train_distributed_v2.py is with the module `torch.distributed.launch` and is simpler for using distributed computing with PyTorch. On node 0, launch it as:

````python
python -m torch.distributed.launch --nnodes 2  --node_rank 0 --nproc_per_node=2 train_distributed_v2.py
````
On node 1, launch it as:
````python
python -m torch.distributed.launch --nnodes 2  --node_rank 1 --nproc_per_node=2 train_distributed_v2.py
````
and so on.
