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
Following the format of the sample file, the preprocess.py will be used to convert the text file into the required data file
# Train
The train.py trains the data, each of which has a size of 1000 x 7
#get Motif
The train.py will keep a trained one. The H5 model file, using this file as well as the data, yields a text file that stores the MOTIF
