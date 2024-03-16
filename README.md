# Robot-Arm-Classification: Project Overview
- Created a convolutional neural network to classify EEG signals to control a UR5e robot arm, serving as a brain-controlled prosthetic.
- Collected 1000 EEG samples for each of the four predetermined gestures and a negative classification using an OpenBCI Electrode Cap. Manually verified and filtered EEGs using a Buttersworth filter.
- Initially explored Decision Trees and various machine learning techniques, resulting in accuracies close to random guessing (~20%). Opted for a CNN due to its effectiveness with spatial data.
- Chose accuracy as the performance metric due to balanced classes and safety considerations for the robot arm. Achieved a model accuracy of 92% with the CNN.
- Despite the promising accuracy, the project was complex and prone to errors, particularly during data collection. The CNN's numerous tunable parameters and signal processing steps required meticulous, isolated parameter tuning for optimization.



# Code and Resources 
**Python Version:** 3.10.12 <br>
**Packages:** NumPy, Matplotlib, Seaborn, OS, Pandas, Keras, SciPy, scikit-learn <br>
**OpenBCI's EEG Electrode Cap:** https://shop.openbci.com/products/openbci-eeg-electrocap <br>
**OpenBCI GUI:** https://docs.openbci.com/Software/OpenBCISoftware/GUIDocs/ <br>
**Universal Robot's UR5e:** https://www.universal-robots.com/products/ur5-robot/<br>



# Data Collection
Over 1000 samples were collected for Gesture 1 using "The OpenBCI GUI". Since the raw EEG data was saved by session, a second person was necessary to annotate the times required to partition them into individual EEGs using [this script](/OpenBCI_GUI_Timing.py). Initially, we believed the GUI was the most efficient way to extrapolate the raw EEG data, but the strain on the computer's battery life was too great. A program running from the machine's terminal shell was [created](/BCI_LiveFeeding.py). We configured it so that the EEGs were saved as individual files, which proved much easier when amassing them into a single database. The program was used for Gestures 2, 3, and 4, as well as the negative classification, Gesture 0.
 <br><br>
The data was too large to upload but has a size of (2980, 250, 16).



# Data Cleaning
Despite each gesture having over 1000 samples, several had to be discarded for a bevy of possible errors stemming from the complexity of EEG collection. Each gesture had each of its samples plotted atop one-another. The resulting ensamble plots acted as a reference that allowed our team to manually differentiate the useable samples from those that were not. [This notebook](/Robot_Arm_Classification,_Data_Cleaning.ipynb) showcases the cleaning of the data. The files in the "...Cuts.txt" format were utilized to section the original EEGs of 625 sample points across 16 channels into 250 sample points across 16 channels. <br>
Due to its collection via The OpenBCI GUI, Gesture 1 required a scale factor comprised of the BCI's resolution and voltage. <br>
In the intrest of class balance, the number of samples for each gesture was limited to the number of samples in Gesture 1, which had 596 approved samples. A such, there were 2980 samples in total.





# Brief eda

# Model building and performance of each (chosen / rejected)

# Insights / perspective gained




    
![Image](/images/BME_CM.png)
