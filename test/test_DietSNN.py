import numpy as np
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
import n3ml.model
import argparse
import torch.nn as nn
import time
import matplotlib.pyplot as plt


class Plot:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax2 = self.ax.twinx()
        plt.title('Diet-SNN')

    def update(self, y1, y2):
        x = np.arange(y1.shape[0]) * 64
        ax1 = self.ax
        ax2 = self.ax2
        ax1.plot(x, y1, 'g')
        ax2.plot(x, y2, 'b')
        ax1.set_xlabel('number of images')
        ax1.set_ylabel('accuracy', color='g')
        ax2.set_ylabel('loss', color='b')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def validate(val_loader, model, criterion):
    total_images = 0
    num_corrects = 0
    total_loss = 0


    for step, (images, labels) in enumerate(val_loader):

        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)
        loss = criterion(preds, labels)

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)

        total_loss += loss.cpu().detach().numpy()* images.size(0)
        total_images += images.size(0)

    val_acc  = num_corrects.float() / total_images
    val_loss = total_loss / total_images
    return val_acc, val_loss



def train(train_loader, model, criterion, optimizer):
    plotter = Plot()

    total_images = 0
    num_corrects = 0
    total_loss = 0
    list_loss = []
    list_acc = []
    for step, (images, labels) in enumerate(train_loader):

        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)

        np.set_printoptions(precision=1)
        o = preds.detach().cpu().numpy()[0]
        print("label: {} - output neuron's voltages: {}".format(labels.detach().cpu().numpy()[0], o))

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
        total_loss   += loss.cpu().detach().numpy() * images.size(0)
        total_images += images.size(0)

        if total_images > 0:  # and total_images % 30 == 0
            list_loss.append(total_loss / total_images)
            list_acc.append(float(num_corrects) / total_images)
            plotter.update(y1=np.array(list_acc), y2=np.array(list_loss))

        for key, value in model.module.leak.items():
            model.module.leak[key].data.clamp_(max=1.0)  # maximum of leak=1.0

    train_acc = num_corrects.float() / total_images
    train_loss = total_loss / total_images
    return train_acc, train_loss

def threshold_balancing(model):
    threshold_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    loader = torch.utils.data.DataLoader(dataset=threshold_dataset, batch_size=64, shuffle=True)

    thresholds = []

    def find(layer):
        max_act = 0

        for step, (images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            output = model(images, find_max_mem=True, max_mem_layer=layer)
            if output > max_act:
                max_act = output.item()

            if step == 0:
                thresholds.append(max_act)
                model.module.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
                break

    for l in model.module.extractor.named_children():
        if isinstance(l[1], nn.Conv2d):
            find(int(l[0]))

    for c in model.module.classifier.named_children():
        if isinstance(c[1], nn.Linear):
            if (int(c[0]) == len(model.module.classifier) - 1):
                break
            else:
                find(int(l[0]) + int(c[0]) + 1)

    return thresholds


def app(opt):
    print(opt)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    model = n3ml.model.Diet2020_SNN(labels=10, timesteps=opt.timesteps, leak=opt.leak, default_threshold=opt.default_threshold, dropout=opt.dropout, kernel_size=opt.kernel_size).cuda()
    model = nn.DataParallel(model)

    pretrained_ann = './pretrained/Diet_ANN.pth'

    if pretrained_ann:
        state = torch.load(pretrained_ann, map_location='cpu')
        cur_dict = model.state_dict()
        for key in state['state_dict'].keys():
            if key in cur_dict:
                if (state['state_dict'][key].shape == cur_dict[key].shape):
                    cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
        model.load_state_dict(cur_dict)
        print('\nstep1. Accuracy of pretrained Diet_ANN model: {}'.format(state['best_acc']))

        thresholds_each_layer = threshold_balancing(model)
        print('\nstep2. Thresholds balancing for each layer: {}'.format(thresholds_each_layer))
        model.module.threshold_update(scaling_factor=opt.scaling_factor, thresholds=thresholds_each_layer[:])


    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
        batch_size=opt.batch_size,
        shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
        batch_size=opt.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=False)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25])
    best_acc = 0

    for epoch in range(opt.num_epochs):
        print('\nstep3. Training Diet_SNN epoch:', epoch)
        start = time.time()
        train_acc, train_loss = train(train_loader, model, criterion, optimizer)
        end = time.time()
        print('total time: {:.2f}s - epoch: {} - accuracy: {} - loss: {}'.format(end - start, epoch, train_acc, train_loss))

        val_acc, val_loss = validate(val_loader, model, criterion)
        if val_acc > best_acc:
            best_acc = val_acc
            print('in test, epoch: {} - best accuracy: {} - loss: {}'.format(epoch, val_acc, val_loss))

        lr_scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('--data',              default='data')
    parser.add_argument('--num_epochs',        default=30,    type=int)
    parser.add_argument('--batch_size',        default=64,    type=int)
    parser.add_argument('--timesteps',         default=20,     type=int)
    parser.add_argument('--leak',              default=1.0,   type=float)
    parser.add_argument('--scaling_factor',    default=0.7,   type=float)
    parser.add_argument('--default_threshold', default=1.0,   type=float)
    parser.add_argument('--dropout',           default=0.2,   type=float)
    parser.add_argument('--kernel_size',       default=3,     type=int)
    parser.add_argument('-lr',                 default=1.0e-4,  type=float)
    parser.add_argument('--momentum',          default=0.95,  type=float)
    parser.add_argument('--weight_decay',      default=5e-4,  type=float)
    app(parser.parse_args())
