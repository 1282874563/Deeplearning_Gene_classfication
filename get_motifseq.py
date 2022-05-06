import keras.models
from keras.models import Model
import numpy as np
import os
import pandas as pd
from keras.models import load_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def get_TPMOBZ():
    x=np.load('D:\\Data\\2W\\allxnew.npy')   ##data(size=N*1000*7)
    y = np.load('D:\\Data\\2W\\alllabel.npy')
    x_P=B73_x[B73_y[:,1]>0]     #####get  positive samples
    return x_P

def get_maxrelu(test_x):
    cnnmodel = keras.models.load_model('D:\\Data\\2W\\7dimmodelATT_TCNnew.h5') #### Load the model file
    cnnmodel.summary()
    desiredOutputs = cnnmodel.get_layer('conv1d_3').output  ####Gets the activation value for the third layer of the convolution layer (the position of the convolution layer can be changed as needed)
    newmodel = Model(cnnmodel.inputs, desiredOutputs)
    reluvalue = newmodel.predict(test_x)
    reluvalue_position=np.argmax(reluvalue,axis=1)   ####Gets the maximum active unit location for the convolutional kernel
    maxrelu=np.amax(reluvalue,axis=1)
    return reluvalue_position,maxrelu
def get_featureseq(reluposition,convlayer):
    y = np.load('D:\\Data\\2W\\alllabel.npy') ###The location of the storage file
    x=np.load(D:\\Data\\2W\\allxnew.npy')  #####Path to store the sequence file
    
    seq=x[y[:,1]>0]  #####get  positive samples
    motif=np.arange(18).reshape(1,18)
   
    for i in range(len(seq)):
        if reluposition[i,convlayer]+18<=10000:
           seqnew=seq[i,B73reluposition[i,convlayer]:B73reluposition[i,convlayer]+18]
           seqnew=seqnew.reshape(1,18)
           for a in seqnew:
               if all(a!='D') and all(a!='I'):
                  motif=np.append(motif,seqnew,axis=0)
    seq=np.delete(seq,0,axis=0)
    return seq
x=get_TPMOBZ()
reluposition,relu=get_maxrelu(x)
seq18=get_featureseq(reluposition,0) ###Get the characteristic sequences of MO17 and B73

np.savetxt("D:\\Data\\2W\\motif18.txt",seq,fmt='%s',delimiter=',')

