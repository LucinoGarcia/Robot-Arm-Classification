```python
import numpy as np
import pandas as pd
import os

from scipy.signal import butter, lfilter

from google.colab import drive
drive.mount('/content/drive')
```


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


```python
print(All_Data.shape)
print(All_Labels.shape)
```

    (2980, 250, 16)
    (2980, 5)
