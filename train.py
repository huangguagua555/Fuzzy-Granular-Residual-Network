import torch
from torch import nn
import torchvision
from torchvision import transforms
from sklearn.metrics import accuracy_score

from torch.utils.data import Dataset,DataLoader,TensorDataset
import pandas as pd
import time
from torchsummary import summary

def train():
    transform = transforms.Compose([transforms.ToTensor()])
    BTACH_SIZE=128
    ds_train = torchvision.datasets.ImageFolder(root="data/tiny-imagenet-200/train",transform=transform)
    ds_valid = torchvision.datasets.ImageFolder(root="data/tiny-imagenet-200/val",transform=transform)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BTACH_SIZE, shuffle=True, num_workers=0)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=BTACH_SIZE, shuffle=False, num_workers=0)

    from models import densenet

    model = densenet.densenet121()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # summary(model, (3, 32, 32))

    def train(model, device, train_loader, optimizer, epoch):
        start = time.time()
        model.train()
        sun_loss = 0
        sum_correct = 0
        step = 0
        for step, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(data)
            loss = model.loss_func(predictions, labels)
            loss.backward()
            optimizer.step()
            sun_loss += loss.item()
            _, preds = predictions.max(1)
            sum_correct += preds.eq(labels).sum()

            if ((step) % 100 == 0) and (step != 0):
                #             print(step)
                #             print(len(data))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                    epoch, step * len(data), len(train_loader.dataset),
                           100. * step / len(train_loader), loss.item() / BTACH_SIZE))

        finish = time.time()
        print((step * len(data), (len(train_loader.dataset))))
        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
        print('Train set: Epoch: {}, Train loss: {:.4f}, Train Accuracy: {:.4f}'.format(
            epoch,
            sun_loss / len(train_loader.dataset),
            sum_correct.float() / (len(train_loader.dataset))))

    def test(model, device, test_loader, epoch):
        start = time.time()
        model.eval()
        test_loss = 0
        correct = 0
        step = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)

                loss = model.loss_func(outputs, labels)
                test_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum()
                step += 1

        finish = time.time()
        print('Test set: Epoch: {}, Val loss: {:.4f}, Val Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss / len(test_loader.dataset),
            correct.float() / len(test_loader.dataset),
            finish - start
        ))
        acc=correct.float() / len(test_loader.dataset)
        return acc

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.loss_func = nn.CrossEntropyLoss()
    EPOCHS = 100
    model.to(device)
    for epoch in range(1, EPOCHS + 1):
        train(model, device, dl_train, optimizer, epoch)
        acc = test(model, device, dl_valid, epoch)
    #     print(df)
    acc = acc.cpu()
    acc = round(float(acc), 5)
    torch.save(model, "./tiny_dense101_acc_" + str(acc) + "_.pt")


if __name__ == '__main__':
    train()












































