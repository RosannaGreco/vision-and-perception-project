import pandas as pd
import csv
#This script makes a csv ('valid_b.csv') containing image paths and respective labels for the first classification
#We have 7 different classes:
#0 elbow
#1 finger
#2 forearm
#3 hand
#4 humerus
#5 shoulder
#6 wrist




#take the lables from image paths

csv_file = './valid_a.csv'
#file containing only image paths


f = open(csv_file)
csvreader = csv.reader(f)

#creating a dictionary
d = {}
rows = []
for row in csvreader:
    
    img_path = row[0]
    path_vec = img_path.split('/')
    label= path_vec[2]
    if (label == 'XR_ELBOW'):
        label = 0
    elif (label == 'XR_FINGER'):
        label = 1
    elif (label == 'XR_FOREARM'):
        label = 2
    elif (label == 'XR_HAND'):
        label = 3
    elif (label == 'XR_HUMERUS'):
        label = 4
    elif (label == 'XR_SHOULDER'):
        label = 5
    elif (label == 'XR_WRIST'):
        label = 6

    d[img_path] = label
f.close()
print(d)


#file which will contain image paths and labels
csv2 = './valid_b.csv'
f2 = open(csv2, 'w')

writer = csv.writer(f2)
for key in d:
    row = [key, d[key]]

    writer.writerow(row)

f2.close()






