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





# Web scrapping / data collection info
# Data cleaning
# Brief eda
# Model building and performance of each (chosen / rejected)
# Insights / perspective gained




    
![Image](/images/BME_CM.png)
