# %%
# Final Project - Kishore Reddy and Akhil Ajikumar - Alexnet based model training.
# %%
#intialising device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %%
# intialising libraries
import torch 
import torchvision
from torchvision import transforms
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



# %%
torch.cuda.empty_cache


# %%
# Data Transformations.

data_transforms = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image brightness
data_jitter_brightness = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.ColorJitter(brightness=-5),
    transforms.ColorJitter(brightness=5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image saturation
data_jitter_saturation = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.ColorJitter(saturation=5),
    transforms.ColorJitter(saturation=-5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image contrast
data_jitter_contrast = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.ColorJitter(contrast=5),
    transforms.ColorJitter(contrast=-5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image hues
data_jitter_hue = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.ColorJitter(hue=0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and rotate image
data_rotate = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image horizontally
data_hflip = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image vertically
data_vflip = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and shear image
data_shear = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomAffine(degrees = 15,shear=2),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and translate image
data_translate = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomAffine(degrees = 15,translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and crop image 
data_center = transforms.Compose([
	transforms.Resize((36, 36)),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and convert image to grayscale
data_grayscale = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# %%
data_transforms = transforms.Compose([transforms.Resize([112,112]),transforms.ToTensor()])

# %%
#initilaising parameters
batch = 256
lr = 0.001
epochs = 8
Classes = 43

# %%
# preparing the dataset
train_path = "/home/akhil/Downloads/GTRSB/Train"
train_data = torchvision.datasets.ImageFolder(root= train_path,transform=data_transforms)

split = 0.8
 
num_train = int(len(train_data)*split)
num_validation = len(train_data) - num_train

train_data,validation_data = data.random_split(train_data,[num_train,num_validation])

print(f"No of Training images = {len(train_data)}")
print(f"No of Validation images = {len(validation_data)}")

# %%

#loading the data

train_load = data.DataLoader(train_data,shuffle=True,batch_size=batch)
validation_load = data.DataLoader(validation_data,shuffle=True,batch_size=batch)


# %%


# %%
def count_param(model):
    return sum(a.numel() for a in model.parameters() if a.requires_grad)

# %%
#importing the model

from class_alexnet import AlexnetTS

model = AlexnetTS(Classes)
print(f'count = {count_param(model)}')



# %%
#initialising optimizer and loss
optimizer = optim.Adam(model.parameters(),lr = lr)
criterion = nn.CrossEntropyLoss()

# %%
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

# %%
print(model)

# %%
# funtion to calculate accuracy
def calculate_accuracy(y_pred,y):
    top_pred = y_pred.argmax(1,keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() /y.shape[0]
    return acc

# %%
# function to train model

def train(model,loader,opt,criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (images,labels) in loader :
        images = images.cuda()
        labels = labels.cuda()
        opt.zero_grad()

        output , _ = model(images)
        loss = criterion(output,labels)
        loss.backward()

        acc = calculate_accuracy(output,labels)

        opt.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss/len(loader),epoch_acc/len(loader)

# %%
#funtion to evaluate model
def evaluate(model,loader,opt,criterion):
    epoch_loss = 0 
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (images,labels) in loader:
            images = images.cuda()
            labels = labels.cuda()

            output,_ = model(images)
            loss = criterion(output,labels)
            
            acc = calculate_accuracy(output,labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss/len(loader),epoch_acc/len(loader)


# %%
# training the model

training_loss_list = [0]*epochs
training_accuracy_list = [0]*epochs
validation_loss_list = [0]*epochs
validation_accuracy_list = [0]*epochs

for epoch in range(epochs):
    print("Epoch - %d : " %(epoch))

    train_start_time = time.monotonic()
    train_loss, train_acc = train (model,train_load,optimizer,criterion)
    train_end_time = time.monotonic()

    validation_start_time = time.monotonic()
    validation_loss,validation_accuracy = evaluate(model,validation_load,optimizer,criterion)
    validation_end_time = time.monotonic()

    training_loss_list[epoch] = train_loss
    training_accuracy_list[epoch] = train_acc
    validation_loss_list[epoch] = validation_loss
    validation_accuracy_list[epoch] = validation_accuracy

    print("Training: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" % (train_loss, train_acc, train_end_time - train_start_time))
    print("Validation: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" % (validation_loss, validation_accuracy, validation_end_time - validation_start_time))
    print("")

# %%
print(torch.cuda.is_available())

# %%
#saving the model 
model_save = "/home/akhil/"
if not os.path.isdir(model_save):
    os.mkdir(model_save)

path_to_model = model_save + "model.pth"
if os.path.exists(path_to_model):
    os.remove(path_to_model)

torch.save(model.state_dict(),path_to_model)




# data plotting
_,axs = plt.subplots(1,2,figsize =(15,5))

axs[0].plot(training_loss_list,label = 'train')
axs[0].plot(validation_loss_list,label ='validation')
axs[0].set_title("Plot-Loss")
axs[0].set_xlabel("epochs")
axs[0].set_ylabel('Loss')
legend = axs[0].legend(loc='upper right',shadow = False)

axs[1].plot(training_accuracy_list,label = 'train')
axs[1].plot(validation_accuracy_list,label ='validation')
axs[1].set_title("Plot-Accuracy")
axs[1].set_xlabel("epochs")
axs[1].set_ylabel('Accuracy')
legend = axs[1].legend(loc='center right',shadow = True)


