# Music-Classifier
In this project, we inplement a music classifier based on GTZAN dataset.
We implement Logistic Regression and Support Vector Machine classifiers
using 3 different feature extraction methods. 
We used Python version 3.5 to run this code. 
We used scikit, librosa, numpy and scipy libraries
We used Spyder IDE to develop code
This code is oonly tested in Ubuntu 16.04 LTE

@Author: Humayra Tasnim & Mohammad Ashraf Siddiquee


To run this code, please install the following libraries with depencies:
	1. Python 3.5
	2. Numpy, Scipy
	3. Scikit learn
	4. Librosa
	
Instructions to run the code: Make sure you have the following elements in same directory:
	1. mc.py (Python source code)
	2. genres (folder containing training wav files)
	3. validation (folder containing testing wav files)
	
To run the code, use the command:
			python mc.py
Make sure your default python in python 3. Otherwise run the following command:
			python3 mc.py
			
The code will execute and genarate retuts.txt file with the testing results.
NB: The audio files should be in WAV format. Files can be converted from AU format to WAV format using SOX command in ubuntu.
	Script for AU to WAV conversion is included in this directory. Put then in specific directories.
	Make sure you have sox installed in Ubuntu.
	
Thank you and good luck!!!
