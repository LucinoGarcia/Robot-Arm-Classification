```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils import to_categorical

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D


from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).


# Gesture Data

### Gesture 1


```python
NewChannelDataRow = 0
G1_Data = np.zeros([911, 625, 16])    # G1 has 911 samples
All_G1 = np.zeros([0, 250, 16])

for session in range(1, 4):
  Rows = pd.read_csv("/content/drive/MyDrive/BME Capstone/Gesture 1/DataBySession/ScriptTimes, Session "+str(session)+".csv").iloc[:, 1]
  Data = pd.read_csv("/content/drive/MyDrive/BME Capstone/Gesture 1/DataBySession/Gesture 1, Session "+str(session)+".csv")

  for row in Rows:
    for Channel in range(16):
      G1_Data[NewChannelDataRow, :, Channel] = Data.iloc[:, Channel][int(row):(int(row)+625)]
    NewChannelDataRow += 1


G1_Data = G1_Data*(8388608/187500)                                     # To scale the Gesture 1 Data into the same range as the other gestures

G1_CutValues = [int(i) for i in list(pd.read_csv("/content/drive/MyDrive/G1_Cuts.txt", header=None).iloc[0, :])]

DropTheseIndex = list(reversed([i for i in range(len(G1_CutValues)) if G1_CutValues[i] == 0]))
for i in DropTheseIndex:
  G1_Data = np.delete(G1_Data, i, axis=0)
  del G1_CutValues[i]

G1_CutValues = [0 if i == 50 else (375 if i > 375 else i) for i in G1_CutValues]


for sample in range(len(G1_Data)):
  All_G1 = np.concatenate((All_G1, [G1_Data[sample, G1_CutValues[sample]:((G1_CutValues[sample])+250), :]]), axis=0)

'''
fig, axs = plt.subplots(4, 4, figsize=(12, 12))

for i in range(4):
    for j in range(4):
        index = i * 4 + j

        for k in range(All_G1.shape[0]):
            axs[i, j].plot(All_G1[k, :, index], color='blue', alpha=0.05)  # Adjust alpha for transparency

        axs[i, j].set_title(f'Plot {index+1}')
        axs[i, j].set_xlabel('Points')
        axs[i, j].set_ylabel('Data')

fig.suptitle('Gesture 1, Segmented')
plt.tight_layout()
plt.show()

del axs, fig, i, j, index, k
'''

del Channel, Data, DropTheseIndex, G1_Data, NewChannelDataRow, G1_CutValues, Rows, i, row, sample, session
```

### Gesture 2


```python
G2_Data = np.zeros([833, 625, 16])    # G2 has 833 samples


for folder, subfolders, files in os.walk(os.path.abspath("/content/drive/MyDrive/BME Capstone/All Gestures/Gesture 2")):
  row = 0
  for i in files:
      Data = pd.read_csv("/content/drive/MyDrive/BME Capstone/All Gestures/Gesture 2/"+i, header=None, dtype=object)
      for Channel in range(16):
        G2_Data[row, :, Channel] = Data.iloc[:, Channel]
      row += 1

G2_CutValues = [int(i) for i in list(pd.read_csv("/content/drive/MyDrive/G2_Cuts.txt", header=None).iloc[0, :])]

DropTheseIndex = list(reversed([i for i in range(len(G2_CutValues)) if G2_CutValues[i] == 0]))
for i in DropTheseIndex:
  G2_Data = np.delete(G2_Data, i, axis=0)
  del G2_CutValues[i]


G2_CutValues = [0 if i == 50 else (375 if i > 375 else i) for i in G2_CutValues]
All_G2 = np.empty((0, 250, 16))

for sample in range(len(G2_Data)):
  All_G2 = np.concatenate((All_G2, [G2_Data[sample, G2_CutValues[sample]:((G2_CutValues[sample])+250), :]]), axis=0)


del Channel, Data, DropTheseIndex, G2_CutValues, G2_Data, files, folder, i, row, sample, subfolders
```

### Gesture 3


```python
G3_Data = np.zeros([980, 625, 16])


for folder, subfolders, files in os.walk(os.path.abspath("/content/drive/MyDrive/BME Capstone/All Gestures/Gesture 3")):
  row = 0
  for i in files:
      Data = pd.read_csv("/content/drive/MyDrive/BME Capstone/All Gestures/Gesture 3/"+i, header=None, dtype=object)
      for Channel in range(16):
        G3_Data[row, :, Channel] = Data.iloc[:, Channel]
      row += 1

G3_CutValues = [int(i) for i in list(pd.read_csv("/content/drive/MyDrive/G3_Cuts.txt", header=None).iloc[0, :])]

DropTheseIndex = list(reversed([i for i in range(len(G3_CutValues)) if G3_CutValues[i] == 0]))
for i in DropTheseIndex:
  G3_Data = np.delete(G3_Data, i, axis=0)
  del G3_CutValues[i]


G3_CutValues = [0 if i == 50 else (375 if i > 375 else i) for i in G3_CutValues]
All_G3 = np.empty((0, 250, 16))

for sample in range(len(G3_Data)):
  All_G3 = np.concatenate((All_G3, [G3_Data[sample, G3_CutValues[sample]:((G3_CutValues[sample])+250), :]]), axis=0)


del Channel, Data, DropTheseIndex, G3_CutValues, G3_Data, files, folder, i, row, sample, subfolders
```

### Gesture 4


```python
G4_Data = np.zeros([1016, 625, 16])


for folder, subfolders, files in os.walk(os.path.abspath("/content/drive/MyDrive/BME Capstone/All Gestures/Gesture 4")):
  row = 0
  for i in files:
      Data = pd.read_csv("/content/drive/MyDrive/BME Capstone/All Gestures/Gesture 4/"+i, header=None, dtype=object)
      for Channel in range(16):
        G4_Data[row, :, Channel] = Data.iloc[:, Channel]
      row += 1


G4_CutValues = [int(i) for i in list(pd.concat([pd.read_csv("/content/drive/MyDrive/G4_Cuts_1.txt", header=None),
                                          pd.read_csv("/content/drive/MyDrive/G4_Cuts_2.txt", header=None).iloc[:, 700:]], axis=1).iloc[0, :])]



DropTheseIndex = list(reversed([i for i in range(len(G4_CutValues)) if G4_CutValues[i] == 0]))
for i in DropTheseIndex:
  G4_Data = np.delete(G4_Data, i, axis=0)
  del G4_CutValues[i]


G4_CutValues = [0 if i == 50 else (375 if i > 375 else i) for i in G4_CutValues]
All_G4 = np.empty((0, 250, 16))

for sample in range(len(G4_Data)):
  All_G4 = np.concatenate((All_G4, [G4_Data[sample, G4_CutValues[sample]:((G4_CutValues[sample])+250), :]]), axis=0)


del Channel, Data, DropTheseIndex, G4_CutValues, G4_Data, files, folder, i, row, sample, subfolders
```

### Gesture 0


```python
G0_Data = np.zeros([502, 625, 16])


for folder, subfolders, files in os.walk(os.path.abspath("/content/drive/MyDrive/BME Capstone/All Gestures/Gesture 0")):
  row = 0
  for i in files:
      Data = pd.read_csv("/content/drive/MyDrive/BME Capstone/All Gestures/Gesture 0/"+i, header=None, dtype=object)
      for Channel in range(16):
        G0_Data[row, :, Channel] = Data.iloc[:, Channel]
      row += 1

All_G0 = np.empty((0, 250, 16))

for sample in range(len(G0_Data)):
  All_G0 = np.concatenate((All_G0, [G0_Data[sample, :250, :], G0_Data[sample, 250:500, :]]), axis=0)


del Channel, Data, G0_Data, files, folder, i, row, sample, subfolders
```

# Filtering and Labeling


```python
def butter_lowpass_filter(data, cutoff_freq=50, sampling_rate=125, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data
```


```python
### Labels

MinSamp = np.min([len(All_G1), len(All_G2), len(All_G3), len(All_G4), len(All_G0)])
All_Data = np.concatenate((butter_lowpass_filter(All_G1[:MinSamp, :, :]), butter_lowpass_filter(All_G2[:MinSamp, :, :]),
                           butter_lowpass_filter(All_G3[:MinSamp, :, :]), butter_lowpass_filter(All_G4[:MinSamp, :, :]),
                           butter_lowpass_filter(All_G0[:MinSamp, :, :])), axis=0)
All_Labels = np.concatenate((np.ones([1, MinSamp]), np.ones([1, MinSamp])*2, np.ones([1, MinSamp])*3, np.ones([1, MinSamp])*4,
                             np.ones([1, MinSamp])*0), axis=1).T

del All_G0, All_G1, All_G2, All_G3, All_G4, MinSamp
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

