import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.autograd import Variable

import torchvision
from torchvision.transforms import Normalize

import json
from sklearn import preprocessing

from utils_gen import sorted_groups

import cv2
import matplotlib.pyplot as plt

import seaborn as sn
import pandas as pd


class Experiments():

    def __init__(self, device, model, train_loader, test_loader, criterion, optimizer):
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.if_multilabel = False
      
    def set_multilabel(self, value=True):
        self.if_multilabel = value
  
    def train(self, epoch_number):
        self.model.train()
        train_loss = 0.
        # get subsequent batches over the data in a given epoch
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # send data tensors to GPU (or CPU)
            data, target = data.to(self.device), target.to(self.device)
            # this will zero out the gradients for this batch
            self.optimizer.zero_grad()
            # this will execute the forward() function
            output = self.model(data)
            # calculate loss using criterion
            loss = self.criterion(output, target)
            # backpropagate the loss
            loss.backward()
            # update the model weights (with assumed learning rate)
            self.optimizer.step()
            train_loss += loss.item()
        print('Train Epoch: {}'.format(epoch_number))
        train_loss /= len(self.train_loader)
        print('\tTrain set: Average loss: {:.4f}'.format(train_loss))
        return train_loss
    
    def train_ae(self, epoch_number):
        self.model.train()
        self.criterion_c = nn.BCELoss(reduction='mean')
        train_loss = 0.
        train_loss_c = 0.
        # get subsequent batches over the data in a given epoch
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # send tensors to GPU (or CPU)
            data, target = data.to(self.device), target.to(self.device)
            # this will zero out the gradients for this batch
            self.optimizer.zero_grad()
            # calculate loss
            if self.model.if_classifer:
                loss, loss_c = self.model._get_reconstruction_and_classification_loss(data, target, self.criterion, self.criterion_c)
                train_loss += loss.item()
                train_loss_c += loss_c.item()
                loss += loss_c
            else:    
                loss = self.model._get_reconstruction_loss(data, self.criterion)
                train_loss += loss.item()
            # backpropagate the loss
            loss.backward()
            # update the model weights (with assumed learning rate)
            self.optimizer.step() 
        #
        print('Train Epoch: {}'.format(epoch_number))
        train_loss /= len(self.train_loader)
        if self.model.if_classifer:
            train_loss_c /= len(self.train_loader)
            print('\tTrain set: Average losses (AE, classifer): {:.4f}, {:.4f}'.format(train_loss, train_loss_c))
        else:
            print('\tTrain set: Average loss: {:.4f}'.format(train_loss))
        return train_loss
    
    def test(self, message=None):
        self.model.eval()
        test_loss = 0.
        correct = 0
        sum_targets = len(self.test_loader.dataset)
        # this is just inference, we don't need to calculate gradients
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device) 
                output = self.model(data)
                # calculate and sum up batch loss
                test_loss += self.criterion(output, target)
                # get the index of class with the max probability
                if self.set_multilabel:
                    prediction = output > .5 
                    prediction = prediction.float()
                    correct += (prediction*target).long().sum().item()  # how many from existing symmeties were detected
                    sum_targets += target.long().sum().item()
                else:
                    prediction = output.argmax(dim=1) 
                    correct += prediction.eq(target).sum().item()
        test_loss /= len(self.test_loader)
        accuracy = correct / sum_targets    
        if message is not None:
            print('\t{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                message, test_loss, correct, sum_targets, 100.*accuracy))
        return test_loss.cpu(), accuracy
    
    def test_ae(self, message=None):
        self.model.eval()
        test_loss = 0.
        # this is just inference, we don't need to calculate gradients
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                # calculate and sum up batch loss
                test_loss += self.model._get_reconstruction_loss(data, self.criterion)
        test_loss /= len(self.test_loader)
        if message is not None:
            print('\t{}: Average loss: {:.4f}'.format(message, test_loss))
        return test_loss.cpu()

    def calculate_confusion(self):
        self.model.eval()
        confusion = np.zeros((self.model.no_classes, self.model.no_classes))
        # this is just inference, we don't need to calculate gradients
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device) 
                output = self.model(data)
                # get the index of class with the max probability 
                prediction = output.argmax(dim=1)  
                for pr, gt in zip(prediction, target):
                    confusion[pr.item(), gt.item()] += 1
        #return confusion.astype(float)/confusion.sum()
        #return confusion/confusion.astype(float).sum(axis=0)
        return  confusion/confusion.astype(float).sum(axis=1, keepdims=True)

    def calculate_confusion_multilabel(self, no_classes):
        self.model.eval()
        confusions = np.zeros((no_classes,2,2))
        # this is just inference, we don't need to calculate gradients
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device) 
                if self.model.if_classifer:
                    _, output = self.model(data)
                else:
                    output = self.model(data)
                prediction = output > .5 
                prediction = prediction.long()
                for pr, gt in zip(prediction, target.long()):
                    for i in range(no_classes):
                        confusions[i, pr[i].item(), gt[i].item()] += 1
        return confusions

    def run_training(self, no_epochs):
        train_loss = []
        validation_loss = []
        validation_accuracy = []
        test_accuracy = []
        for epoch_number in range(1, no_epochs+1):
            train_loss.append(self.train(epoch_number))
            val_loss, val_acc = self.test('Validation set')
            validation_loss.append(val_loss)
            validation_accuracy.append(val_acc)
        # and select test accuracy for the best epoch (with the highest validation accuracy)
        best_accuracy = np.amax(validation_accuracy)
        return train_loss, validation_loss, best_accuracy
    
    def run_training_ae(self, no_epochs):
        train_loss = []
        validation_loss = []
        for epoch_number in range(1, no_epochs+1):
            train_loss.append(self.train_ae(epoch_number))
            val_loss = self.test_ae('Validation set')
            validation_loss.append(val_loss)
        return train_loss, validation_loss


class CustomDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, data_dir, file_paths, labels, normalize=False, downsample=True):
        self.data_dir = data_dir
        self.file_paths = file_paths
        self.labels = labels
        self.normalize = normalize
        self.transform = Normalize(.5,.5)
        self.downsample = downsample
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]
        # Read an image with OpenCV
        image = cv2.imread(os.path.join(self.data_dir, file_path), cv2.IMREAD_GRAYSCALE)
        if image.dtype == 'uint8': image = image.astype('float32')/255.  # return to float
        image = 1. - image
        # naive downsampling
        if self.downsample:
            image = image[::2,::2]
        if self.normalize:
            return self.transform(torch.tensor(image[None,...])), torch.tensor(label).type((torch.float32))
        else:
            return torch.tensor(image[None,...]), torch.tensor(label).type((torch.float32))

def parse_dataset_classification(dataset_directory, labels='labels.json', label_dict=sorted_groups, splitratio=[.8,.2], batchsize=128, normalize=False, downsample=True):
    label_file = open(os.path.join(dataset_directory, labels))
    label_data = json.load(label_file)
    datafiles=[]
    datalabels=[]
    for ld in label_data:
        datafiles.append(str(ld["file_id"])+'_f.png')
        datalabels.append(ld["label"])
    #le = preprocessing.LabelEncoder()
    #datalabels = le.fit_transform(datalabels)
    datalabels = [label_dict[dl] for dl in datalabels]  # translate respective groups into numbers
    dataset = CustomDataset(dataset_directory, datafiles, datalabels, normalize=normalize, downsample=downsample)
    trainsize = int(len(dataset)*splitratio[0])
    train_set, val_set = random_split(dataset, [trainsize, len(dataset)-trainsize])
    train_loader = DataLoader(train_set, batch_size=batchsize, 
                               shuffle=True, drop_last=True, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batchsize)
    return train_loader, val_loader

def plot_loss(train_loss, validation_loss, title, logscale=False):
    plt.grid(True)
    plt.xlabel("subsequent epochs")
    plt.ylabel('average loss')
    if logscale: plt.yscale('log')
    plt.plot(range(1, len(train_loss)+1), train_loss, 'o-', label='training')
    plt.plot(range(1, len(validation_loss)+1), validation_loss, 'o-', label='validation')
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join('./', 'loss.png'), bbox_inches='tight', dpi=200)
    plt.close()

def save_loss(train_loss, validation_loss):
    np.savetxt("loss.txt", np.c_[train_loss, validation_loss])

def plot_confusion(confusion_matrix):
    class_labels = {v: k for k, v in sorted_groups.items()}
    class_labels = [class_labels[i] for i in range(len(class_labels))]
    df_cm = pd.DataFrame(confusion_matrix, index = class_labels, columns = class_labels)
    plt.figure(figsize = (20,16))
    ax = sn.heatmap(df_cm, annot=True)
    ax.set(xlabel="ground truth", ylabel="prediction")
    plt.savefig(os.path.join('./', 'confusion.png'), bbox_inches='tight', dpi=200)
    plt.close()
    
def print_confusion(confusion_matrix, symmetry_names):
    print("no | symmetry name | precision | recall")
    for i in range(confusion_matrix.shape[0]):
        cm = confusion_matrix[i]
        print("{0:2d} | {1:13s} | {2:1.5f}   | {3:1.5f}".format(i, symmetry_names[i], 
                                                                      cm[1,1]/(cm[1,1]+cm[1,0]), 
                                                                      cm[1,1]/(cm[1,1]+cm[0,1])))


class Deep(nn.Module):
    # this defines the structure of the Perceptron model
    def __init__(self, no_classes=2):
        super(Deep, self).__init__()
        self.no_classes = no_classes
        # fully connected layers
        self.fc1 = nn.Linear(500*500, 200)
        self.fc2 = nn.Linear(200, 40)
        self.fc3 = nn.Linear(40, no_classes)

    def forward(self, x):
        x = x.view(-1, 500*500)
        x = self.fc1(x)
        x = F.relu(x)
        # hidden layers
        x = self.fc2(x)
        x = F.relu(x)
        # classification layer
        x = self.fc3(x)
        return F.log_softmax(x, dim=1) # note that dim=0 is the number of samples in batch


class SimpleCNN(nn.Module):
    '''
    simple CNN model
    '''
    def __init__(self, no_classes=2):
        super(SimpleCNN, self).__init__()
        self.no_classes = no_classes
        self.conv0 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv1 = nn.Conv2d(16, 16, 3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=3)
        self.conv4 = nn.Conv2d(64, 16, 3, stride=3)
        self.fc1 = nn.Linear(500*500, 200)
        self.fc2 = nn.Linear(200, 40)
        self.fc3 = nn.Linear(40, no_classes)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


VGG_TYPES = {'vgg11' : torchvision.models.vgg11, 
             'vgg11_bn' : torchvision.models.vgg11_bn, 
             'vgg13' : torchvision.models.vgg13, 
             'vgg13_bn' : torchvision.models.vgg13_bn, 
             'vgg16' : torchvision.models.vgg16, 
             'vgg16_bn' : torchvision.models.vgg16_bn,
             'vgg19_bn' : torchvision.models.vgg19_bn, 
             'vgg19' : torchvision.models.vgg19}

class Custom_VGG(nn.Module):

    def __init__(self, 
                 pretrained=True, 
                 ipt_size=[100,100],
                 vgg_type='vgg11', 
                 no_classes=10,
                 if_classifer=True,
                 if_multilabel=False,
                 latent_dim=128):
        super(Custom_VGG, self).__init__()
        vgg_loader = VGG_TYPES[vgg_type]
        vgg = vgg_loader(pretrained=pretrained)
        self.features = vgg.features
        self.no_classes = no_classes
        # init fully connected part of vgg
        test_ipt = Variable(torch.zeros(1,3,ipt_size[0],ipt_size[1]))
        test_out = vgg.features(test_ipt)
        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.latent_dim = latent_dim
        if if_classifer:
            self.classifier = nn.Sequential(nn.Linear(self.n_features, 512),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            nn.Linear(512, self.no_classes))
            if if_multilabel:
                self.classifier.append(nn.Sigmoid())
            else:
                self.classifier.append(nn.LogSoftmax(dim=1))
        else:
            self.classifier = nn.Sequential(nn.Linear(self.n_features, self.latent_dim))
        self._init_classifier_weights()

    def forward(self, x):
        x = x.expand(-1,3,-1,-1)  # expanding to 3 channels
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_classifier_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class SiameseCNN(nn.Module):
    '''
    simple CNN model
    '''
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2d(1, 6, 5)
        self.conv12 = nn.Conv2d(6, 16, 5)
        self.conv13 = nn.Conv2d(16, 16, 5)
        self.conv21 = nn.Conv2d(1, 6, 5)
        self.conv22 = nn.Conv2d(6, 16, 5)
        self.conv23 = nn.Conv2d(16, 16, 5)
        self.pool = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(16 * 5 * 5 * 2, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)

    def singlepass(self, x):
        x = x[:,0]
        x = self.pool(F.relu(self.conv11(x)))
        x = self.pool(F.relu(self.conv12(x)))
        x = self.pool(F.relu(self.conv13(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.log_softmax(self.fc3(x))
        x = F.relu(self.fc3(x))
        return x
    
    def forward(self, x):
        x1 = self.singlepass(x[:,0])
        x2 = self.singlepass(x[:,1])
        x3 = self.singlepass(x[:,2])
        return torch.stack([x1,x2,x3], dim=0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

vgg_cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg11s': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 256, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vgg22': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 256, 'M', 512, 512, 512, 512, 512, 'M', 512, 512, 512, 512, 512, 'M']
}

def DoubleVGG11():
    model = VGG(make_layers(vgg_cfg['vgg11s']), make_layers(vgg_cfg['vgg11s']))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(8*16*c_hid, latent_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return x
    
class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 8*16*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.Sigmoid() # The input images is scaled between 0 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 8, 8)
        x = self.net(x)
        return x
    
class Autoencoder(nn.Module):

    def __init__(self,
                 base_channel_size: int = 16,
                 latent_dim: int = 64,
                 num_input_channels: int = 3,
                 if_classifer=False,
                 no_classes=None,
                 device = None):
        super().__init__()
        self.if_classifer = if_classifer
        self.no_classes = no_classes
        # Creating encoder and decoder
        #self.encoder = Encoder(num_input_channels, base_channel_size, latent_dim).to(device)
        self.encoder = Custom_VGG(pretrained=True, 
                                  ipt_size=[250,250], 
                                  vgg_type='vgg11', 
                                  if_classifer=False,
                                  latent_dim=latent_dim)
        if device is not None: 
            self.encoder.to(device)
        self.decoder = Decoder(num_input_channels, base_channel_size, latent_dim)
        if device is not None: 
            self.decoder.to(device)
        if self.if_classifer:
            self.classifier = nn.Sequential(nn.Linear(latent_dim, 512),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(True),
                                            nn.Dropout(),
                                            nn.Linear(512, self.no_classes))
            self.classifier.append(nn.Sigmoid())
            self._init_classifier_weights()
            if device is not None: 
                self.classifier.to(device)

    def parameters(self):
        if self.if_classifer:
            return list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.classifier.parameters()) 
        else:        
            return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def train(self):
        self.encoder.train()
        self.decoder.train()
        if self.if_classifer:
            self.classifier.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        if self.if_classifer:
            self.classifier.eval()
        
    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        if self.if_classifer:
            return x_hat, self.classifier(z)
        else:
            return x_hat
    
    def _get_reconstruction_loss(self, x, loss_F):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        if self.if_classifer:
            x_hat, _ = self.forward(x)
        else:
            x_hat = self.forward(x)
        loss = loss_F(x, x_hat)
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss
        
    def _get_reconstruction_and_classification_loss(self, x, target, loss_F, loss_c_F):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x_hat, c = self.forward(x)
        loss = loss_F(x, x_hat)
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss, loss_c_F(c, target)

    def _init_classifier_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()