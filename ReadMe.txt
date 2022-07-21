Our project consists in three classifications done with a CNN and a ResNet architectures. 

The MURA dataset can be downloaded from  here : https://stanfordmlgroup.github.io/competitions/mura/

It contains seven different classes, containing several kinds of anomalies. 
The first classificator will return the kind of bone the image is displaying, the second will tell us if there are any 
anomalies, the third performs an anomaly classification on a particular class of our dataset. 
Running the main.py and giving as input an image path, it is possible to see the models 
and test them on an image.


$ python3 main.py -i <image_path>


The third classificator has been trained only on humerus images, so, if the image is not an humerus
and doesn't contain any anomalies, the third prediction will be considered not available. 
In order to test the third classificator properly, it is possible also to use 
TestAnomalyClassificator.py, which will plot a random image from the humerus anomalies dataset we 
created and show the corresponding prediction and the ground truth. 

Some of the first model we trained is available in the Models Folder. The second and the third can be
downloaded using the link in the file Models.txt in the Models Folder. 
But it is also possible to train the nets again using the 
training.py file, which uses the functions defined in Train.py. 
The models can be trained in this way: 

$ python3 training.py -first true

$ python3 training.py -second true

$ python3 training.py -third true


In the Train.py script there are also commented lines testing a different kind of loss. 
For the second classification, we obtained the best results using Negative Log Likelihood loss,
but we tested the model also using Cross Entropy Loss. 

The MuraDataset.py script contains the definition of a class used to define our dataset. We applied
also invertion and histogram equalization to the images. 

In the Useful_Scripts folders there are the scripts used to obtain the csv files used for the first
and second classifications. As we wrote above, the third classificator has been 
trained only on humerus images, so we classificated some of the images 
(humerous_anomalies.csv) and used the DataAugmentation.py script to create more data 
applying some operations and to add paths and labels to a csv file (humerous_anomalies_final.csv)

The implemented models are defined in CNN.py and ResNet.py. 

Accuracy.py contains a function computing the accuracy of our models. 
