# Final Project - Kishore Reddy and Akhil Ajikumar - Alexnet based model training.

# loading the libraries
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from torchvision import datasets,models,transforms
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


# data transform
test_transforms = transforms.Compose([transforms.Resize([112,112]),transforms.ToTensor()])

# Loading dataset
test_data_path = "/home/akhil/Downloads/GTRSB/Test"
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=test_transforms)
test_loader = data.DataLoader(test_data,batch_size=1,shuffle=False)



numClasses = 43

# printing the labels
num = range(numClasses)
labels = []
for i in num:
    labels.append(str(i))
labels = sorted(labels)
for i in num:
    labels[i] = int(labels[i])
print("List of labels : ")
print("Actual labels \t--> Class in PyTorch")
for i in num:
    print("\t%d \t--> \t%d" % (labels[i], i))

# reading the class file
df = pd.read_csv("/home/akhil/Downloads/GTRSB/Test.csv")
numExamples = len(df)
labels_list = list(df.ClassId)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# loading the model
from class_alexnet import AlexnetTS
MODEL_PATH = "/home/akhil/Downloads/model.pth"
model = AlexnetTS(numClasses)

num_ftrs = model.fc.in_features
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 43)
model = model.to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.cuda()


# predicting the data
y_pred_list = []
corr_classified = 0

with torch.no_grad():
    model.eval()

    i = 0

    for image, _ in test_loader:
        image = image.cuda()

        y_test_pred = model(image)

        y_pred_softmax = torch.log_softmax(y_test_pred[0], dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        y_pred_tags = y_pred_tags.cpu().numpy()
        
        y_pred = y_pred_tags[0]
        y_pred = labels[y_pred]
        
        y_pred_list.append(y_pred)

        if labels_list[i] == y_pred:
            corr_classified += 1

        i += 1

print("Number of correctly classified images = %d" % corr_classified)
print("Number of incorrectly classified images = %d" % (numExamples - corr_classified))
print("Final accuracy = %f" % (corr_classified / numExamples))

# classification
print(classification_report(labels_list, y_pred_list))

# printing confusion matrix
def plot_confusion_matrix(labels, pred_labels, classes):
    
    fig = plt.figure(figsize = (20, 20))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, display_labels = classes)
    cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
    plt.xticks(rotation = 20)
    
labels_arr = range(0, numClasses)
plot_confusion_matrix(labels_list, y_pred_list, labels_arr)


