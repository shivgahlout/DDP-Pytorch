import os
import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.models as models
# import pydicom


def get_dataset(batch_size):

    transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456, 0.456, 0.456], std=[0.224, 0.224, 0.224])])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, \
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,\
        sampler=train_sampler)

    return train_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = '19.16.19.19' ##enter server address here
    os.environ['MASTER_PORT'] = '8888'

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.manual_seed(0)

    torch.cuda.set_device(args.local_rank)

    model=models.resnet18(pretrained=True)
    model.fc=nn.Linear(512, 10)
    model.cuda(args.local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])


    batch_size = 64
    criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    train_loader=get_dataset( batch_size)




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
            if args.local_rank == 0:
                print('Epoch [{}/{}],Loss: {}'.format(epoch + 1, 10,loss.item()))






if __name__ == '__main__':
    main()
