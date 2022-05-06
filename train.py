import keras.optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add, Input, Conv1D, Activation, Flatten, Dense,BatchNormalization,MaxPooling1D,Dropout,Bidirectional,GRU
from tensorflow.keras import regularizers
import numpy as np
from tensorflow.keras import initializers
from tensorflow.keras.layers import Multiply
from tensorflow.python.keras.layers.core import *
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
#  residuals
def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu',kernel_initializer=initializers.RandomNormal(),kernel_regularizer=regularizers.l2(0.0002))(x)  # 第一卷积
    r=BatchNormalization()(r)
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate,activation='relu')(r)  
    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)  # shortcut
    o = add([r, shortcut])
    o = Activation('relu')(o)  
    return o
## Attention module
def attention_3d_block(inputs,SINGLE_ATTENTION_VECTOR):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)    ##It’s equivalent to transposing
    a = Dense(inputs.shape[1], activation='softmax')(a)   ###Calculate weights for each time step
    if SINGLE_ATTENTION_VECTOR==1:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)###Average each dimension
        a = RepeatVector(input_dim)(a)  #### To the same dimension
    a_probs = Permute((2, 1))(a)### Transpose
    output_attention_mul = Multiply()([inputs, a_probs])####Multiplicative weight
    return output_attention_mul
#  Sequence classification model
def ATT_TCN_BiGRU(train_x, train_y, valid_x, valid_y):
    inputs = Input(shape=(10000,7))
    x = ResBlock(inputs, filters=24, kernel_size=10, dilation_rate=1)
    x = ResBlock(x, filters=36, kernel_size=3, dilation_rate=3)
    x = ResBlock(x, filters=36, kernel_size=3, dilation_rate=9)
    x = ResBlock(x, filters=36, kernel_size=3, dilation_rate=27)
    x = MaxPooling1D(10)(x)
    # x=LSTM(units=30,dropout=0.1,return_sequences=True)(x)
    x = Bidirectional(GRU(units=100,activation='tanh', recurrent_initializer='orthogonal',return_sequences=True,dropout=0.1),merge_mode='concat')(x)  # -------
    x = attention_3d_block(x, 1)
    x = Flatten()(x)
    x = Dense(2,activation='softmax')(x)
    model = Model(inputs, x)
    #  View network structure
    model.summary()
    # Compilation model
    from keras import optimizers
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # # Training Model
    from keras.callbacks import ModelCheckpoint
    filepath = 'D:\\Data\\2W\\modelATT_TCNnew.h5'####The storage location of the model
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_accuracy',
                                 verbose=0,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1
                                 )
    callbacks_list = [checkpoint]
    cost=model.fit(train_x, train_y, batch_size=16, epochs=30
                   , verbose=1, validation_data=(valid_x, valid_y),callbacks=callbacks_list)
    #Plotting
    valloss = cost.history['val_loss']
    train_loss = cost.history['loss']
    valacc = cost.history['val_accuracy']
    train_acc = cost.history['accuracy']
    import matplotlib.pyplot as plt
    plt.plot(valacc, color='g', label='val_acc')
    plt.plot(train_acc, color='r', label='train_acc')
    plt.title('val_acc and train_acc')
    plt.legend()
    plt.show()
    plt.plot(valloss, color='g', label='val_loss')
    plt.plot(train_loss, color='r', label='train_loss')
    plt.title('val_loss and train loss')
    plt.legend()
   
    
train_x,val_x,train_y,val_y=train_test_split(x,y,test_size=0.1)   
ATT_TCN_BiGRU(train_x,train_y,val_x,val_y)
################ Calculate AUC
model=keras.models.load_model('D:\\Data\\2W\\modelATT_TCNew.h5')
print(roc_auc_score(val_y,model.predict(val_x)))
