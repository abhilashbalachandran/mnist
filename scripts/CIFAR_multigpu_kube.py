import os
#set 1,2,3 gpu
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#progress bar
import tqdm


#torch ddp modules
import torch.multiprocessing as mp # used to spawn multiple processes in multiple GPUS -> one in each GPU

from torch.utils.data.distributed import DistributedSampler  # used for chunking the data across GPUS with non overlapping data (set shuffle to false)
from torch.nn.parallel import DistributedDataParallel as DDP #used to distributing the model replicas
from torch.distributed import init_process_group, destroy_process_group 


def ddp_setup(rank, world_size):
    '''
    Args:
        rank: unique id assigned to each process (assigned by torch mp)
        world_size: total num processes to be spawned in a group
    '''
    # os.environ['MASTER_ADDR'] = "localhost" #running master on localhost
    # os.environ["MASTER_PORT"] = "12355" # any free port
    # print(f"os environ variables MASTER_ADDR = {os.environ.get('MASTER_ADDR')}, MASTER_PORT = {os.environ.get('MASTER_PORT')}")
    init_process_group(backend="nccl", rank=rank, world_size=world_size) #nccl = nvidia collective communications library
    torch.cuda.set_device(0)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Trainer:
    def __init__(
            self,
            model: torch.nn.Module, 
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
            world_size: int,

    )-> None:
        self.gpu_id = 0
        self.model = model.to('cuda:0')
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids = [0])
        self.world_size = world_size

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def _run_epoch(self,epoch):
        batch_size = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        total_samples = 0
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            epoch_loss+=loss.item()
            total_samples += source.size(0)  # Accumulate the number of samples processed
        #aggregate loss from all processes
        epoch_loss_tensor = torch.tensor(epoch_loss).to(self.gpu_id)
        total_samples_tensor = torch.tensor(total_samples).to(self.gpu_id)
        torch.distributed.all_reduce(epoch_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)
        # Compute average loss
        average_loss = epoch_loss_tensor.item() / total_samples_tensor.item()   
        if self.gpu_id == 0:  # Typically, we let the master process do the printing
            print(f"Average epoch loss: {average_loss*1e2:.4f}")

            
    
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict() #notice model.module
        PATH = "data/cifar_net_ddp.pth"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id==0 and epoch%self.save_every==0: #only save model from master process. Since they are synchronized, it should be same across all processes
                self._save_checkpoint(epoch)


def load_train_objs(transform=None):
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return train_set, model, optimizer 

def load_test_objs(transform=None):
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    return test_set



def perpare_dataloader(dataset: Dataset, batch_size: int, pin_memory=False, shuffle=False, type="train"):
    if type=="train":
        return DataLoader(dataset, batch_size=batch_size,
                        shuffle=shuffle, pin_memory=pin_memory, sampler=DistributedSampler(dataset))
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)




class Tester:
    def __init__(
            self,
            model: torch.nn.Module,
            test_data: DataLoader,
            device:str,
            classes: list = None
    )->None:
        self.model = model.to(device)
        self.test_data = test_data
        self.device = device
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] if classes is None else classes
    def evaluate(self, 
                 input):
        return self.model(input)
    def calculate_total_accuracy(self):
        correct = 0 
        total = 0
        with torch.no_grad():
            for data in self.test_data:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.evaluate(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    def calculate_accuracy_per_class(self):
        # prepare to count predictions for each cMASTER_ADDRlass
        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}
        with torch.no_grad():
            for data in self.test_data:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.evaluate(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1
        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def main(rank:int, world_size:int, save_every: int, total_epochs: int, batch_size: int):
    #setup ddp and train data
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs(transform=transform)

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    train_data = perpare_dataloader(dataset=dataset, batch_size=batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every, world_size=world_size)
    trainer.train(total_epochs)
    destroy_process_group()

def evaluate(batch_size):
    #evaluate on test data
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = load_test_objs(transform)
    test_dataset = perpare_dataloader(dataset=test_dataset, batch_size=batch_size, type="test")
    #load model
    model = Net()
    model.load_state_dict(torch.load("/data/cifar_net_ddp.pth"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
    print(f"using gpu = {device}")
    tester = Tester(model=model, test_data=test_dataset,device=device)
    tester.calculate_total_accuracy()
    tester.calculate_accuracy_per_class()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model',default=32)
    parser.add_argument('--save_every', type=int, help='How often to save a snapshot',default=5)
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    print(f"is cuda available = {torch.cuda.is_available()}")
    print(f"num gpus available = {torch.cuda.device_count()}")
    
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    print(f"using world size = {world_size}")
    main(rank, world_size, args.save_every, args.total_epochs, args.batch_size)
    # mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
    # evaluate(args.batch_size)