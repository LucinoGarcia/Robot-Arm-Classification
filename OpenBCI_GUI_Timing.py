from IPython import get_ipython
from datetime import datetime
get_ipython().magic('clear')


StartTimes = []                                                         # empty list for start of EEGs of interest 
stopper = ''                                                            # reference variable

while stopper == '' or stopper == 'n':
    StartTimes.append(datetime.utcnow().strftime('%H:%M:%S.%f')[:-3])   # add time (as a string) into StartTimes list
    if stopper == 'n':                                                  # 'n' will be clicked if bad EEG
        StartTimes.append('Bad EEG')                                    # note bad EEG
        stopper == ''                                                   # keeps the program looping
    stopper = input('Click Enter: ')                                    # click enter to keep looping the script
    



#%%                 Saving the data



# set directory

now = datetime.now()
RightNow = now.strftime("%d-%m, %H-%M-%S")                              # saves current time as a string

RoughTimes = open("Rough Times - "+str(RightNow)+".txt", 'w')           # creates a writable txt file
for time in StartTimes:                                                 # goes through each string in the StartTimes list
    RoughTimes.write(time+', Gesture 1\n')                              # writes time, gesture number, and starts new line in txt file
RoughTimes.close()                                                      # saves and closes the txt file

print('Your file has been saved')                                       # message to EEG tTechnician
