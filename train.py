import torch
import time
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader
from dataset import load_mnist, load_cifar10, load_fashion_mnist, imshow
from model import LeNet5

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

trainset, testset = load_mnist()
trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
testloader = DataLoader(testset, batch_size=16, shuffle=False)


@torch.no_grad()
def accuracy(model, data_loader):
    model.eval()
    correct, total = 0, 0
    for batch in data_loader:
        images, labels = batch
        images = transforms.Grayscale()(images)
        images = transforms.Resize(28)(images)
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total


model = LeNet5.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# loop over the dataset multiple times
num_epoch = 8
model.train()
start_time = time.time()
for epoch in range(num_epoch):
    epoch_start_time=time.time()
    running_loss = 0.0
    for i, batch in enumerate(trainloader, 0):
        # get the images; batch is a list of [images, labels]
        images, labels = batch
        images = transforms.Grayscale()(images)
        images = transforms.Resize(28)(images)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # zero the parameter gradients

        # get prediction
        outputs = model(images)
        # compute loss
        loss = loss_fn(outputs, labels)
        # reduce loss
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:  # print every 1000 mini-batches
            train_acc = accuracy(model, trainloader)
            test_acc = accuracy(model, testloader)
            model.train()
            # print('[epoch %2d, batch %5d] loss: %.3f' %
            #       (epoch + 1, i + 1, running_loss / 500))
            print('[epoch %2d, batch %5d] loss: %.3f | train acc: %.3f | test acc: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500,100 * train_acc,100 * test_acc))
            running_loss = 0.0
    epoch_end_time=time.time()
    print("epoch %d finished, time:%.3f"%(epoch+1,epoch_end_time-epoch_start_time))
finish_time=time.time()
total_time= finish_time-start_time
print("total time:", total_time)
model_file = 'model.pth'
torch.save(model.state_dict(), model_file)
print(f'Model saved to {model_file}.')
print('Finished Training')

# show some prediction result
dataiter = iter(testloader)
# images, labels = dataiter.next()
images, labels = next(dataiter)
images = images.to(device)
predictions = model(images).argmax(1).detach().cpu()

classes = trainset.classes
print('GroundTruth: ', ' '.join('%5s' % classes[i] for i in labels))
print('Prediction: ', ' '.join('%5s' % classes[i] for i in predictions))
imshow(torchvision.utils.make_grid(images.cpu()))

# test

# train_acc = accuracy(model, trainloader)
test_acc = accuracy(model, testloader)
# print('Accuracy on the train set: %f %%' % (100 * train_acc))
print('Accuracy on the test set: %f %%' % (100 * test_acc))
