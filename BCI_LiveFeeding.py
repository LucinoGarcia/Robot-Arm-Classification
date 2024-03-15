
from pyOpenBCI import OpenBCICyton
import numpy as np
from datetime import datetime


currentEEG = np.zeros([625, 16])
rowNum = 0



def print_raw(sample):
    global rowNum
    currentEEG[rowNum] = sample.channels_data
    rowNum += 1
    if rowNum == 625:
        board.stop_stream()





board = OpenBCICyton(port='/dev/cu.usbserial-DM0258JH', daisy=True)

print('Start thinking')
board.start_stream(print_raw)
print('Stop thinking')


RightNow = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
np.savetxt("Potential EEG - "+str(RightNow)+".txt", currentEEG, delimiter=',')


print('Your file has been saved. Be sure to thank Lucino')