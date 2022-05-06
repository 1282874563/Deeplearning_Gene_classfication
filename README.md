# Deeplearning_Gene_classfication
In this study, we obtained a reliable deep learning model based on CNN and RNN, which can help to predict the corn annular RNA sequence.
The training data is composed of the sequences of B73 and MO17, two different corn materials.
Tere are 3w of them,For the interpretability of the model, each level of convolution kernel in CNN structure can be regarded as motif scanner, and we applied visual method to find the motif that the model learned.
For more details, users can refer to the published article.
# Requirements 
This model uses the Keras framework. You need to run the code in the following environment.


*Python 3.7


*Keras 2.4.3


*Pandas 1.2.4


*tensorflow 2.3.0


*numpy 1.19.0
# Preprocess
   By using this model, First Following the format of the exsample file to prepare your own raw data and annotation file,
Then change the filepath in preprocess.py and run this file.It will be used to convert the text file into the required data file（.npy file）
# Train
 change the (.npy file)filepath in train.py. Then run train.py to train the data, each of which has a size of 10000 x 7. It will creat amodel file(.h5) 
# get Motif
 Input the H5 model file's layer name and data file(.npy), It will creat a text file that stores the MOTIF
# Exsample file
We uploaded the exsample file（Annotation exsample.txt and data exsample.txt） to help you better prepare the raw data file and raw Annotation file  to use our code.
Especially we upload exsample data file(.npy) and label file(.npy) which can be used to train,you can download exsample.rar.

