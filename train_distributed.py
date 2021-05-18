import os
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.models as models



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_nodes', default=1, type=int, help='total number of the nodes')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--node_rank', default=0, type=int, help='number of node (server) usig for traning')
    args = parser.parse_args()


    args.world_size = args.gpus * args.total_nodes
    os.environ['MASTER_ADDR'] = '192.168.29.199'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))



def get_dataset(world_size, global_rank, batch_size):

    transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456, 0.456, 0.456], std=[0.224, 0.224, 0.224])])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=world_size,rank=global_rank)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, \
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,\
        sampler=train_sampler)

    return train_loader


def train(local_rank, args):
    ## global_rank is the global rank of the process (among all the GPUs (not just on a particular node). )
    global_rank = args.node_rank * args.gpus + local_rank
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=global_rank)
    torch.manual_seed(0)
    torch.cuda.set_device(local_rank)
   
    batch_size = 64

    criterion = nn.CrossEntropyLoss().cuda(local_rank)

    model=models.resnet18(pretrained=True)
    model.fc=nn.Linear(512, 10)
    model.cuda(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train_loader=get_dataset(args.world_size, global_rank, batch_size)
    
    for epoch in range(10):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ## if this condition is not used, output will print for each process (GPU)
            if local_rank == 0:
                print('Epoch [{}/{}],Loss: {}'.format(epoch + 1, 10,loss.item()))


if __name__ == '__main__':
    main()
