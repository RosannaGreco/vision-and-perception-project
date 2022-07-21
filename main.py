# Imports
from CNN import CNN
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torchinfo import summary
from argparse import ArgumentParser
import torchvision.transforms as tf
from Accuracy import check_accuracy
import PIL.ImageOps
import torchvision
from torch.utils.data import DataLoader
from MuraDataset import MuraDataset
from ResNet import ResNet152
from Train import bone_classification, anomaly_detection, anomaly_classification


labels_map_CL1 = {
    0: "Elbow",
    1: "Finger",
    2: "Forearm",
    3: "Hand",
    4: "Humerus",
    5: "Shoulder",
    6: "Wrist",
}

labels_map_CL2 = {
    0: "Anomaly not detected",
    1: "Anomaly detected",
}

labels_map_CL3 = {
    0: "Fracture",
    1: "Screws and plates",
    2: "Other anomalies"
}


def GreatestNumberIndex(tensor, n):
    greatest = -20
    greatest_idx = 0
    for i in range(0, n):
        if tensor[i] > greatest:
            greatest = tensor[i]
            greatest_idx = i
        i += 1
    return greatest_idx


parser = ArgumentParser()
parser.add_argument('-i', '--image', required=True, help='path to image directory')
parser.add_argument('-p', '--plot', default=True, help='plot also the image')
args = parser.parse_args()

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LOAD MODELS FOR ALL SEQUENCE ----------------------------------------------------
# 1st classification
# load model
model_cl1 = CNN(7)
model_cl1.load_state_dict(torch.load('./Models/CNN.pt'))
model_cl1.eval()
print('-------------------------- MODEL LOADED --------------------------')
# print model properties
print(summary(model_cl1, input_size=([5, 3, 32, 32])))

# 2nd classification
# load model
model_cl2 = ResNet152(3, 2)
model_cl2.load_state_dict(torch.load('./Models/RESNET_ANOMALIES_DETECTION_NLL.pt'))
model_cl2.eval()
print('-------------------------- MODEL LOADED --------------------------')
# print model properties
print(summary(model_cl2, input_size=([48, 3, 64, 64])))

# 3rd classification
# load model
model_cl3 = ResNet152(3, 3)
model_cl3.load_state_dict(torch.load('./Models/RESNET_ANOMALIES_CLASSIFICATION.pt'))
model_cl3.eval()
print('-------------------------- MODEL LOADED --------------------------')
# print model properties
print(summary(model_cl3, input_size=([5, 3, 64, 64])))


img = Image.open(args.image)
img = img.resize((32, 32))
img = img.convert('RGB')
image = PIL.ImageOps.invert(img)
image = torchvision.transforms.functional.equalize(image)
transform = tf.ToTensor()
image = transform(image)
image_plot = tf.ToPILImage()(image)
image = torch.unsqueeze(image, 0)

res_cl1 = model_cl1(image)
res_cl1_t = res_cl1[0]
res_cl1 = GreatestNumberIndex(res_cl1_t, 7)
res_cl1 = labels_map_CL1[res_cl1]

res_cl2 = model_cl2(image)
res_cl2_t = res_cl2[0]
res_cl2 = GreatestNumberIndex(res_cl2_t, 2)
res_cl2 = labels_map_CL2[res_cl2]

if res_cl1 == 'Humerus' and res_cl2 == 'Anomaly detected':
    res_cl3 = model_cl3(image)
    res_cl3_t = res_cl3[0]
    res_cl3 = GreatestNumberIndex(res_cl3_t, 3)
    res_cl3 = labels_map_CL3[res_cl3]

else:
    res_cl3 = 'No prediction available for the selected image'

fig = plt.figure(figsize=(15, 15))
plt.imshow(img)
plt.title(f'Bone classificator: {res_cl1} \n 'f'Anomaly detector: {res_cl2} \n 'f'Anomaly classificator: {res_cl3} ')
plt.axis('off')
plt.show()







