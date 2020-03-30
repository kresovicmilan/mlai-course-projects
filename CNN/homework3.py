#Milan Kresovic - Erasmus student s266915
#This code was run on Google colab
%matplotlib inline
import torch
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.optim as optim
import torchvision.datasets as datasets

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np

'''Variables'''
n_classes = 100
n_epochs = 20
'''
0 - fully connected Neural Netowrk 
{batch_size = 256, num_epochs = 20, 32x32 resolution, learning_rate = 0.0001}

1 - CNN with convolutional filters 32/32/32/64
{batch_size = 256, num_epochs = 20, 32x32 resolution, learning_rate = 0.0001}

2 - CNN with convolutional filters 128/128/128/256
{batch_size = 256, num_epochs = 20, 32x32 resolution, learning_rate = 0.0001}

3 - CNN with convolutional filters 256/256/256/512
{batch_size = 256, num_epochs = 20, 32x32 resolution, learning_rate = 0.0001}

4 - CNN with convolutional filters 512/512/512/1024 (slow training)
{batch_size = 256, num_epochs = 20, 32x32 resolution, learning_rate = 0.0001}

5 - homworkPart == 2, with batch normalization on every layer
{batch_size = 256, num_epochs = 20, 32x32 resolution, learning_rate = 0.0001}

6 - homworkPart == 6, with first fully connected layer wider (8192 neurons)
{batch_size = 256, num_epochs = 20, 32x32 resolution, learning_rate = 0.0001}

7 - homeworkPart == 6, with dropout 0.5 on FC1 (4096 neurons)
{batch_size = 256, num_epochs = 20, 32x32 resolution, learning_rate = 0.0001}

8 - homeworkPart == 2, with random horizontal flipping (data augmentation)
{batch_size = 256, num_epochs = 20, 32x32 resolution, learning_rate = 0.0001}

9 - homeworkPart == 2, with random crop (data augmentation)
{batch_size = 256, num_epochs = 20, 32x32 resolution, learning_rate = 0.0001}

10 - homeworkPart == 2, with random crop (data augmentation)
{batch_size = 256, num_epochs = 20, 32x32 resolution, learning_rate = 0.0001}

11 - ResNet18, with best data augmentation schema
{batch_size = 128, num_epochs = 10, 224x224 resolution, learning_rate = 0.0001}'''
homeworkPart = 0


# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def plot_kernel(model):
    model_weights = model.state_dict()
    fig = plt.figure()
    plt.figure(figsize=(10,10))
    for idx, filt  in enumerate(model_weights['conv1.weight']):
    #print(filt[0, :, :])
        if idx >= 32: continue
        plt.subplot(4,8, idx + 1)
        plt.imshow(filt[0, :, :], cmap="gray")
        plt.axis('off')
    
    plt.show()

def plot_kernel_output(model,images):
    fig1 = plt.figure()
    plt.figure(figsize=(1,1))
    
    img_normalized = (images[0] - images[0].min()) / (images[0].max() - images[0].min())
    plt.imshow(img_normalized.numpy().transpose(1,2,0))
    plt.show()
    output = model.conv1(images)
    layer_1 = output[0, :, :, :]
    layer_1 = layer_1.data

    fig = plt.figure()
    plt.figure(figsize=(10,10))
    for idx, filt  in enumerate(layer_1):
        if idx >= 32: continue
        plt.subplot(4,8, idx + 1)
        plt.imshow(filt, cmap="gray")
        plt.axis('off')
    plt.show()

def plotLossAccuracy(LossEpochList, AccuracyEpochList, n_epochs):
  epochsList = np.arange(n_epochs)
  
  plt.figure(figsize= (14, 7))
  plt.subplot(1, 2, 1)
  plt.plot(epochsList, LossEpochList, label = "Loss")
  plt.xticks(epochsList)
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.grid()
  
  plt.subplot(1, 2, 2)
  plt.plot(epochsList, AccuracyEpochList, label = "Accuracy")
  plt.xticks(epochsList)
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.grid()
  
  plt.tight_layout()

  plt.show()
  
  
  
def test_accuracy(net, dataloader):
  ########TESTING PHASE###########
  
    #check accuracy on whole test set
    correct = 0
    total = 0
    net.eval() #important for deactivating dropout and correctly use batchnorm accumulated statistics
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test set: %d %%' % (accuracy))
    return accuracy

# function to define an old style fully connected network (multilayer perceptrons)
class old_nn(nn.Module):
    def __init__(self):
        super(old_nn, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, n_classes) #last FC for classification 

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
      
      
#function to define the convolutional network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #conv2d first parameter is the number of kernels at input (you get it from the output value of the previous layer)
        #conv2d second parameter is the number of kernels you wanna have in your convolution, so it will be the n. of kernels at output.
        #conv2d third, fourth and fifth parameters are, as you can read, kernel_size, stride and zero padding :
        conv1In, conv1Out, conv2In, conv2Out, conv3In, conv3Out, convFinIn, convFinOut, numNeuro = setParameters()
        self.conv1 = nn.Conv2d(conv1In, conv1Out, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(conv2In, conv2Out, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(conv3In, conv3Out, kernel_size=3, stride=1, padding=0)
        
        if homeworkPart in [5, 6, 7]:
          self.bn = nn.BatchNorm2d(conv1Out, track_running_stats=False)     
          self.bnLast = nn.BatchNorm2d(convFinOut, track_running_stats=False)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_final = nn.Conv2d(convFinIn, convFinOut, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(convFinOut * 4 * 4, numNeuro)
        
        if homeworkPart == 7:
            self.dropout = nn.Dropout2d(0.5)
        
        self.fc2 = nn.Linear(numNeuro, n_classes) #last FC for classification 
        
    def forward(self, x):
        if homeworkPart in [5, 6, 7]:
          x = F.relu(self.bn(self.conv1(x)))
          x = F.relu(self.bn(self.conv2(x)))
          x = F.relu(self.bn(self.conv3(x)))
          x = F.relu(self.pool(self.bnLast(self.conv_final(x))))
        else:
          x = F.relu(self.conv1(x))
          x = F.relu(self.conv2(x))
          x = F.relu(self.conv3(x))
          x = F.relu(self.pool(self.conv_final(x)))
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        
        if homeworkPart == 7:
          x = self.dropout(x)
        
        x = self.fc2(x)
        return x

"""Setting parameters values for different parts of the homework"""
def setParameters():
  if homeworkPart == 1:
    conv1In = 3
    conv1Out = 32
    conv2In = conv1Out
    conv2Out = 32
    conv3In = conv2Out
    conv3Out = 32
    convFinIn = conv3Out
    convFinOut = 64
    numNeuro = 4096
  elif homeworkPart == 2:
    conv1In = 3
    conv1Out = 128
    conv2In = conv1Out
    conv2Out = 128
    conv3In = conv2Out
    conv3Out = 128
    convFinIn = conv3Out
    convFinOut = 256
    numNeuro = 4096
  elif homeworkPart == 3:
    conv1In = 3
    conv1Out = 256
    conv2In = conv1Out
    conv2Out = 256
    conv3In = conv2Out
    conv3Out = 256
    convFinIn = conv3Out
    convFinOut = 512
    numNeuro = 4096
  elif homeworkPart == 4:
    conv1In = 3
    conv1Out = 512
    conv2In = conv1Out
    conv2Out = 512
    conv3In = conv2Out
    conv3Out = 512
    convFinIn = conv3Out
    convFinOut = 1024
    numNeuro = 4096
  elif homeworkPart == 5:
    conv1In = 3
    conv1Out = 128
    conv2In = conv1Out
    conv2Out = 128
    conv3In = conv2Out
    conv3Out = 128
    convFinIn = conv3Out
    convFinOut = 256
    numNeuro = 4096
  elif homeworkPart == 6:
    conv1In = 3
    conv1Out = 128
    conv2In = conv1Out
    conv2Out = 128
    conv3In = conv2Out
    conv3Out = 128
    convFinIn = conv3Out
    convFinOut = 256
    numNeuro = 8192
  elif homeworkPart == 7:
    conv1In = 3
    conv1Out = 128
    conv2In = conv1Out
    conv2Out = 128
    conv3In = conv2Out
    conv3Out = 128
    convFinIn = conv3Out
    convFinOut = 256
    numNeuro = 4096
  elif homeworkPart == 8:
    conv1In = 3
    conv1Out = 128
    conv2In = conv1Out
    conv2Out = 128
    conv3In = conv2Out
    conv3Out = 128
    convFinIn = conv3Out
    convFinOut = 256
    numNeuro = 4096
  elif homeworkPart == 9:
    conv1In = 3
    conv1Out = 128
    conv2In = conv1Out
    conv2Out = 128
    conv3In = conv2Out
    conv3Out = 128
    convFinIn = conv3Out
    convFinOut = 256
    numNeuro = 4096
  elif homeworkPart == 10:
    conv1In = 3
    conv1Out = 128
    conv2In = conv1Out
    conv2Out = 128
    conv3In = conv2Out
    conv3Out = 128
    convFinIn = conv3Out
    convFinOut = 256
    numNeuro = 4096
  
  return conv1In, conv1Out, conv2In, conv2Out, conv3In, conv3Out, convFinIn, convFinOut, numNeuro
    
   ####RUNNING CODE FROM HERE:

if homeworkPart == 8:
  transform_train = transforms.Compose(
    [
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
  
  transform_test = transforms.Compose(
    [
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
elif homeworkPart == 9:
  transform_train = transforms.Compose(
    [
    transforms.Resize((40,40)),
    transforms.RandomCrop((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
  
  transform_test = transforms.Compose(
    [
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
elif homeworkPart == 10:
  transform_train = transforms.Compose(
    [
    transforms.RandomHorizontalFlip(),
    transforms.Resize((40,40)),
    transforms.RandomCrop((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
  
  transform_test = transforms.Compose(
    [
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
elif homeworkPart == 11:
  #These values of mean, and std were used because of this link: https://pytorch.org/docs/stable/torchvision/models.html
  transform_train = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),
     transforms.Resize((256,256)),
     transforms.RandomCrop((224,224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])
  
  transform_test = transforms.Compose(
    [
     transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
     ])
  
else:
  #transform are heavily used to do simple and complex transformation and data augmentation
  transform_train = transforms.Compose(
    [
     transforms.Resize((32,32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

  transform_test = transforms.Compose(
    [
     transforms.Resize((32,32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])

if homeworkPart == 11:
  trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                          download=False, transform=transform_train)
  
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4,drop_last=True)

  testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                         download=False, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4,drop_last=True)
  
else:
  trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                            shuffle=True, num_workers=4,drop_last=True)

  testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                         download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                           shuffle=False, num_workers=4,drop_last=True)


dataiter = iter(trainloader)

###OPTIONAL:
# show images just to understand what is inside the dataset ;)
#images, labels = dataiter.next()
#imshow(torchvision.utils.make_grid(images))
####


#create the old style NN network
#net = old_nn()
###
if homeworkPart == 0:
  net = old_nn()
elif homeworkPart == 11:
  net = models.resnet18(pretrained=True)
  net.fc = nn.Linear(512, n_classes)
else:
  net = CNN()

####
#for Residual Network:
#net = models.resnet18(pretrained=True)
#net.fc = nn.Linear(512, n_classes) #changing the fully connected layer of the already allocated network
####

###OPTIONAL:
#print("####plotting kernels of conv1 layer:####")
#plot_kernel(net)
####


net = net.cuda()

criterion = nn.CrossEntropyLoss().cuda() #it already does softmax computation for use!
optimizer = optim.Adam(net.parameters(), lr=0.0001) #better convergency w.r.t simple SGD :)


###OPTIONAL:
#print("####plotting output of conv1 layer:#####")
#plot_kernel_output(net,images)  
###

########TRAINING PHASE###########
n_loss_print = len(trainloader)  #print every epoch, use smaller numbers if you wanna print loss more often!

#List of values of accuracy by each epoch
AccuracyEpochList = np.zeros(n_epochs)
#List of values of loss by each epoch
LossEpochList = np.zeros(n_epochs)

#loop over the dataset multiple times
for epoch in range(n_epochs):
    
    #important for activating dropout and correctly train batchnorm
    net.train()
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        # get the inputs and cast them into cuda wrapper
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % n_loss_print == (n_loss_print -1):    
            #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / n_loss_print))
            LossEpochList[epoch] = running_loss / n_loss_print
            running_loss = 0.0

    accuracy = test_accuracy(net,testloader)
    AccuracyEpochList[epoch] = accuracy
    print("----------\nEpoch %d\nLoss: %.3f\nAccuracy: %.3f\n" % (epoch + 1, LossEpochList[epoch], AccuracyEpochList[epoch]))


print('\nFinished Training')

plotLossAccuracy(LossEpochList, AccuracyEpochList, n_epochs)
