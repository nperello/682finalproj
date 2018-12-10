from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class UTKFaceDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.labels_frame.iloc[idx, 0])
        image = io.imread(img_name)
        labels = self.labels_frame.iloc[idx, 1:].as_matrix()
        labels = labels.astype('float')#.reshape(-1, 2)
        image = np.asarray(image)/255
        sample = {'image': image, 'labels': labels}
        if self.transform:
            sample = self.transform(sample)


        return sample, img_name

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}

dataset = UTKFaceDataset(csv_file='UTKfacesalignrace.csv',
                        root_dir='alignedimgs/',
                        transform=transforms.Compose([ToTensor()]))

datasetB = UTKFaceDataset(csv_file='UTKfacesaligngenderB.csv',
                        root_dir='alignedimgs/',
                        transform=transforms.Compose([ToTensor()]))
print(dataset[0])


batch_size = 50
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
# Creating PT data samplers and loaders:
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

print(len(train_loader)*batch_size)
print(len(validation_loader)*batch_size)
for i_batch, sample_batched in enumerate(train_loader):
        print(i_batch, sample_batched)
        break

def show_landmarks_batch(sample_batched):
    images_batch, labels_batch = \
            sample_batched['image'], sample_batched['labels']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    print("labels ", sample_batched['labels'])
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

doShow = False
if doShow:
    for i_batch, sample_batched in enumerate(train_loader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['labels'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

import torch.nn as nn
import torch.nn.functional as F

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(
            nn.Linear(3*3*384, 512),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc3 = nn.Sequential(
            nn.Linear(512, 5),
            nn.ReLU())

    def forward(self, x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = F.interpolate(out, size=(3, 3), mode='bilinear')
        #print(out.shape)
        out = out.view(out.size(0),-1)
        #print(out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return F.log_softmax(out)

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 40
num_classes = 5
learning_rate = 0.0001

model = ConvNet().to(device)
model.apply(init_weights)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def check_accuracy(loader, model):
    
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for data in loader:
            #print(data)
            names = data[1]
            data = data[0]
            #print(data)
            #print()
            x = data['image'].float()
            #print(x)
            #print()
            x= x.to(device=device, dtype=torch.float)  # move to device, e.g. GPU
            #print("BUG")
            #print(data['labels'])
            y = data['labels'].long()
            y = y.view(y.numel())
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            #print(scores)
            _, preds = scores.max(1)
            #print(preds)
            #print(names)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train_model(model, optimizer, criterion, device, num_epochs=1):
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, x in enumerate(train_loader):
            model.train()
            names = x[1]
            x = x[0]
            #if i == 0:
                #print(x)
            images = x['image'].float()
            labels= x['labels'].long()
            labels = labels.view(labels.numel())
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            #print(labels)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                check_accuracy(validation_loader, model)
                running_loss = 0.0


train_model(model, optimizer, criterion, device, num_epochs)

check_accuracy(validation_loader, model)
