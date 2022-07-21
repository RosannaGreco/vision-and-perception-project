from Train import bone_classification, anomaly_classification, anomaly_detection
from CNN import CNN
from ResNet import ResNet152
from argparse import ArgumentParser
import torch

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = ArgumentParser()
parser.add_argument('-first', '--first', default=False, help='train of first classificator')
parser.add_argument('-second', '--second', default=False, help='train of second classificator')
parser.add_argument('-third', '--third', default=False, help='train of third classificator')
args = parser.parse_args()

if args.first:
    print('--------------------------1st CLASSIFIER TRAINING: recognizing body parts--------------------------')
    n_classes = 7
    model = CNN(n_classes)
    print('-------------------------- TRAINING THE MODEL --------------------------')
    bone_classification(model)

if args.second:
    print('--------------------------2nd CLASSIFIER TRAINING: recognizing anomalies--------------------------')
    n_classes = 2
    model2 = ResNet152(3, n_classes)
    print('-------------------------- TRAINING THE MODEL --------------------------')
    anomaly_detection(model2)

if args.third:
    print('--------------------------3nd CLASSIFIER TRAINING: classifying anomalies--------------------------')
    n_classes = 4
    model3 = ResNet152(3, n_classes)
    print('-------------------------- TRAINING THE MODEL --------------------------')
    anomaly_classification(model3)

exit(0)
