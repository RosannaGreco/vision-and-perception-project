#This script makes a csv containing the image paths and the labels for the second classification.
#The label is 1 if an anomaly is detected, 0 otherwise



import pandas as pd
import csv

# file containing image paths

csv_file = './valid_a.csv'


f = open(csv_file)
csvreader = csv.reader(f)


d = {}
rows = []
for row in csvreader:
    #rows.append(row)
    img_path = row[0]
    path_vec = img_path.split('/')
    label= path_vec[4]
    if (label == 'study1_positive'):
        label = 1
    elif (label == 'study2_positive'):
        label = 1
    elif (label == 'study3_positive'):
        label = 1
    elif (label == 'study4_positive'):
        label = 1
    elif (label == 'study1_negative'):
        label = 0
    elif (label == 'study2_negative'):
        label = 0
    elif (label == 'study3_negative'):
        label = 0
    elif (label == 'study4_negative'):
        label = 0

    d[img_path] = label
f.close()
print(d)


#the second file will contain image paths and corresponding lables
csv2 = './valid_c.csv'
f2 = open(csv2, 'w')

writer = csv.writer(f2)
for key in d:
    row = [key, d[key]]

    writer.writerow(row)

f2.close()






