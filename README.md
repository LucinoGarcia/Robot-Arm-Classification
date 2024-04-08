# Robot-Arm-Classification: Project Overview
- Created a convolutional neural network to classify EEG signals to control a UR5e robot arm, serving as a brain-controlled prosthetic.
- Collected 1000 EEG samples for each of the four predetermined gestures and a negative classification using an OpenBCI Electrode Cap. Manually verified and filtered EEGs using a Buttersworth filter.
- Initially explored Decision Trees and various machine learning techniques, resulting in accuracies close to random guessing (~20%). Opted for a CNN due to its effectiveness with spatial data.
- Chose accuracy as the performance metric due to balanced classes and safety considerations for the robot arm. Achieved a model accuracy of 92% with the CNN.
- Despite the promising accuracy, the project was complex and prone to errors, particularly during data collection. The CNN's numerous tunable parameters and signal processing steps required meticulous, isolated parameter tuning for optimization.



# Resources 
**Python Version:** 3.10.12 <br>
**Packages:** NumPy, Matplotlib, Seaborn, OS, Pandas, Keras, SciPy, scikit-learn <br>
**OpenBCI's EEG Electrode Cap:** https://shop.openbci.com/products/openbci-eeg-electrocap <br>
**OpenBCI GUI:** https://docs.openbci.com/Software/OpenBCISoftware/GUIDocs/ <br>
**Universal Robot's UR5e:** https://www.universal-robots.com/products/ur5-robot/<br>



# Data Collection
Over 1000 samples were collected for Gesture 1 using "The OpenBCI GUI". Since the raw EEG data was saved by session, a second person was necessary to annotate the times required to partition them into individual EEGs using [this script](/OpenBCI_GUI_Timing.py). Initially, we believed the GUI was the most efficient way to extrapolate the raw EEG data, but the strain on the computer's battery life was too great. A program running from the machine's terminal shell was [created](/BCI_LiveFeeding.py). We configured it so that the EEGs were saved as individual files, which proved much easier when amassing them into a single database. The program was used for Gestures 2, 3, and 4, as well as the negative classification, Gesture 0.
 <br><br>
The raw data was too large to upload but had a size of (4744, 625, 16).



# Data Cleaning
Despite each gesture having over 1000 samples, several had to be discarded for a bevy of possible errors stemming from the complexity of EEG collection. Each gesture had its samples plotted atop one another. The resulting ensemble plots acted as a reference that allowed our team to manually differentiate between usable samples and those that were not. [This notebook](/Robot_Arm_Classification,_Data_Cleaning.ipynb) showcases the cleaning of the data. The files in the "...Cuts.txt" format were utilized to segment the original EEGs of 625 sample points into 250 sample points, due to the BCI's sampling frequency of 125 Hz. <br><br>
Due to its collection via The OpenBCI GUI, Gesture 1 required a scale factor comprised of the BCI's resolution and voltage. <br><br>
In the interest of class balance, the number of samples for each gesture was limited to the number of samples in Gesture 1, which had 596 approved samples. As such, the cleaned data had a size of (2980, 250, 16).



# EDA
Due to the volatility of raw EEG data, neither measure of dispersion nor central tendancy of will prove very insightful. Ensemble plots can better showcase EEG behavior. Below is ensemble plots of Gesture 3 by channel. The other ensembles can be found in [this project's EDA notebook](/Robot_Arm_Classification,_EDA.ipynb).

![Image](/images/G3_Ens.png)

While class imbalance isn't too much of a problem until the majority class is 5x greater than the minority, that rule-of-thumb only really applies to larger datasets. The image below displays the number of samples per gesture.

![Image](/images/Number_of_Samples_by_Gesture.png)



# Model Building
With each sample having 4000 data points and being 3-dimensional, the complexity and time constraints left few options. Rudimentary CNN, LSTM, SVM, GBM, and Random Forest models were built, but only the CNN and SVM models yielded accuracies significantly greater than a random guess. Due to our team's lack of familiarity with SVMs and the aforementioned time constraints, the CNN was chosen as the model to be further refined. <br><br>
Accuracy was chosen as the model's verification metric due to its interpretability and safety precautions regarding the UR5e Robot Arm. <br><br>
The CNN model is relatively simple with two convolutional layers, each followed by a max-pooling layer, a 128-node ReLU Dense layer, and a final 5-node SoftMax Dense layer for classification. [The CNN model](/Robot_Arm_Classification_CNN.hdf5) yielded a fairly high accuracy of 92%. The notebook for the model can be [found here](/Robot_Arm_Classification,_Model_Building.ipynb). Below is the confusion matrix of the model.

    
![Image](/images/BME_CM.png)



# Potential Implications
- There are a plethora of ways EEG data collection can go awry, ranging from the electrical interference of phones in the room to the subject's mood and stress level during that session.
- The CNN likely performed well with the data due to its ability to handle spatial data.
- The SVM could have possibly performed even better due to its proficiency in classification with higher dimensional data and dimension reduction capability.
- The model's high accuracy of 92% is likely due to the fact that the data was collected from, and therefore overfit to, a single person.
<br><br><br>


[GitHub Repository for Brain-Controlled Prosthetic Gesture Classification](https://github.com/LucinoGarcia/Robot-Arm-Classification)
[Back to Portfolio](https://lucinogarcia.github.io/Portfolio/)
