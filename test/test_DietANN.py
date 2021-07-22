import torch
import time
import torch.nn as nn
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms

import os
import n3ml.model


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

        total_loss += loss.item()* images.size(0)
        total_images += images.size(0)

    val_acc  = num_corrects.float() / total_images
    val_loss = total_loss / total_images
    return val_acc, val_loss


def train(train_loader, model, criterion, optimizer):

    total_images = 0
    num_corrects = 0
    total_loss = 0

    for step, (images, labels) in enumerate(train_loader):

        images = images.cuda()
        labels = labels.cuda()

        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_corrects += torch.argmax(preds, dim=1).eq(labels).sum(dim=0)
        total_loss += loss.item() * images.size(0)
        total_images += images.size(0)

    train_acc = num_corrects.float() / total_images
    train_loss = total_loss / total_images
    return train_acc, train_loss


def app(opt):
    print(opt)

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
        num_workers=8,
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
        batch_size=opt.batch_size, num_workers=8)


    model = n3ml.model.Diet2020_ANN(dropout=opt.dropout).cuda()
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr,  momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[135, 210, 270]) # 0.45, 0.70, 0.90
    best_acc = 0

    for epoch in range(opt.num_epochs):
        print('\n epoch:', epoch)
        start = time.time()
        train_acc, train_loss = train(train_loader, model, criterion, optimizer)
        end = time.time()
        print('total time: {:.2f}s - epoch: {} - accuracy: {} - loss: {}'.format(end - start, epoch, train_acc, train_loss))

        val_acc, val_loss = validate(val_loader, model, criterion)

        if val_acc > best_acc:
            best_acc = val_acc
            state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()}

            try:
                os.mkdir('./pretrained')
            except OSError:
                pass
            torch.save(state, opt.pretrained)
            print('in test, epoch: {} - best accuracy: {} - loss: {}'.format(epoch, val_acc, val_loss))

        lr_scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',        default='data')
    parser.add_argument('--num_epochs',  default=300,   type=int)
    parser.add_argument('--batch_size',  default=64,    type=int)
    parser.add_argument('--lr',          default=1e-02, type=float)
    parser.add_argument('--pretrained',  default='pretrained/Diet_ANN.pth')
    parser.add_argument('--dropout',     default=0.2,   type=float)
    parser.add_argument('--kernel_size', default=3,     type=int)
    app(parser.parse_args())
