```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils import to_categorical

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D
```



# DNN


```python
All_Data = np.load("All_Data.npy")
All_Labels = np.load("All_Labels.npy")
```


```python
All_Labels = to_categorical(All_Labels, num_classes=5)
In_Train, In_Test, Out_Train, Out_Test =  train_test_split(All_Data, All_Labels, test_size = 0.2, random_state=321)

# CNN accepts 3 dimentional data
In_Train = In_Train.reshape(In_Train.shape[0], 250, 16, 1)
In_Test = In_Test.reshape(In_Test.shape[0], 250, 16, 1)
```


```python
CNN = Sequential()

CNN.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape = In_Train[0].shape))

CNN.add(MaxPool2D(pool_size=(2, 2)))
CNN.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

CNN.add(MaxPool2D(pool_size=(2, 2)))
CNN.add(Flatten())

CNN.add(Dense(units=128, activation='relu'))

CNN.add(Dense(units=5, activation='softmax'))

CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


CNN.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 248, 14, 32)       320       
                                                                     
     max_pooling2d (MaxPooling2  (None, 124, 7, 32)        0         
     D)                                                              
                                                                     
     conv2d_1 (Conv2D)           (None, 122, 5, 64)        18496     
                                                                     
     max_pooling2d_1 (MaxPoolin  (None, 61, 2, 64)         0         
     g2D)                                                            
                                                                     
     flatten (Flatten)           (None, 7808)              0         
                                                                     
     dense (Dense)               (None, 128)               999552    
                                                                     
     dense_1 (Dense)             (None, 5)                 645       
                                                                     
    =================================================================
    Total params: 1019013 (3.89 MB)
    Trainable params: 1019013 (3.89 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________



```python
trainingEpochs = CNN.fit(In_Train, Out_Train, batch_size=32, epochs=7, validation_split=0.1)
```

    Epoch 1/7
    68/68 [==============================] - 24s 325ms/step - loss: 227680.9062 - accuracy: 0.7469 - val_loss: 1763.7485 - val_accuracy: 0.7699
    Epoch 2/7
    68/68 [==============================] - 13s 188ms/step - loss: 2263.5847 - accuracy: 0.7972 - val_loss: 787.1883 - val_accuracy: 0.7657
    Epoch 3/7
    68/68 [==============================] - 17s 245ms/step - loss: 1075.6500 - accuracy: 0.8000 - val_loss: 493.6936 - val_accuracy: 0.7573
    Epoch 4/7
    68/68 [==============================] - 14s 210ms/step - loss: 1893.7679 - accuracy: 0.7786 - val_loss: 383.0010 - val_accuracy: 0.7197
    Epoch 5/7
    68/68 [==============================] - 14s 205ms/step - loss: 2986.1743 - accuracy: 0.7762 - val_loss: 884.4761 - val_accuracy: 0.7531
    Epoch 6/7
    68/68 [==============================] - 18s 270ms/step - loss: 1085.7446 - accuracy: 0.7776 - val_loss: 1759.1813 - val_accuracy: 0.7071
    Epoch 7/7
    68/68 [==============================] - 13s 189ms/step - loss: 1041.6101 - accuracy: 0.7678 - val_loss: 686.6398 - val_accuracy: 0.7071



```python
y_pred = np.argmax(CNN.predict(In_Test), axis = 1)
Out_Test = np.argmax(Out_Test, axis=1)
cm = confusion_matrix(Out_Test, y_pred)
print(cm)
```

    19/19 [==============================] - 3s 163ms/step
    [[103   0   0   3  10]
     [  0 111   0   1   0]
     [  0  52  65   0   0]
     [  2  27   1  92   0]
     [  5  58   4   1  61]]

