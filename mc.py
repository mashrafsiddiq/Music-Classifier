
# =============================================================================
# In this project, we inplement a music classifier based on GTZAN dataset.
# We implement Logistic Regression and Support Vector Machine classifiers
# using 3 different feature extraction methods. 
# We used Python version 3.5 to run this code. 
# We used scikit, librosa, numpy and scipy libraries
# We used Spyder IDE to develop code
# 
# @Author: Humayra Tasnim & Mohammad Ashraf Siddiquee
# 
# =============================================================================


import os
from scipy.io import wavfile
from scipy import fft
import numpy as np
import sys
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import librosa
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

# This variable defines the feature extraction mode
# 1 means FFT, 2 means MFCC, 3 means Custom
feature_mode = 2

# This variable defines the classifier mode
# 1 means Logistic Regression, 2 means Support Vector Machine
classifier_mode = 2

# The list of 4 genres
genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# Names of 100 validation files
validation_files = ["validation.00596",  "validation.02436",  "validation.02930",  "validation.03364",  "validation.03550",  "validation.04625",  "validation.07561",  "validation.07642",  "validation.09065",  "validation.09885",  "validation.12150",  "validation.13350",  "validation.16676",  "validation.17817",  "validation.18392",  "validation.18794",  "validation.18828",  "validation.19082",  "validation.21589",  "validation.23193",  "validation.23211",  "validation.23301",  "validation.24620",  "validation.25407",  "validation.26050",  "validation.26581",  "validation.28540",  "validation.28558",  "validation.29001",  "validation.29659",  "validation.30482",  "validation.30768",  "validation.32510",  "validation.32787",  "validation.33033",  "validation.33529",  "validation.34800",  "validation.34835",  "validation.34938",  "validation.36126",  "validation.39797",  "validation.41870",  "validation.43948",  "validation.44485",  "validation.47303",  "validation.49246",  "validation.51339",  "validation.51932",  "validation.52165",  "validation.52376",  "validation.53339",  "validation.54199",  "validation.56037",  "validation.60352",  "validation.62461",  "validation.62917",  "validation.64717",  "validation.65355",  "validation.66836",  "validation.67898",  "validation.68246",  "validation.68504",  "validation.68600",  "validation.69645",  "validation.70325",  "validation.70514",  "validation.70521",  "validation.70709",  "validation.71178",  "validation.71558",  "validation.72738",  "validation.72918",  "validation.73325",  "validation.73749",  "validation.76427",  "validation.76541",  "validation.77219",  "validation.79401",  "validation.80325",  "validation.80446",  "validation.80480",  "validation.81197",  "validation.84038",  "validation.86647",  "validation.87593",  "validation.87812",  "validation.88839",  "validation.89176",  "validation.89749",  "validation.91278",  "validation.91579",  "validation.93577",  "validation.94066",  "validation.94863",  "validation.95021",  "validation.95763",  "validation.96483",  "validation.97080",  "validation.97638",  "validation.97760"]

# Extract feature using FFT
def get_fft_feat(loc):
    sample_rate, X = wavfile.read(loc)
    fft_features = abs(fft(X)[:1000])
    return fft_features

# Extract feature using MFCC
def get_mfcc_feat(loc):
    y, sr = librosa.load(loc)
    feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=80)
    feat = feat.mean(axis=1) + feat.std(axis=1)
    return feat

# Extract feature using Custom Method
def get_custom_feat(loc):
    y, sr = librosa.load(loc)
    feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=80)
    feat = feat.mean(axis=1) + feat.std(axis=1)
    sample_rate, X = wavfile.read(loc)
    fft_features = abs(fft(X)[:1000])
    return feat + fft_features

# Create confusion matrix
def get_confusion_matrix(act, pred):
    return confusion_matrix(act, pred)

# Initiliza logistic regression classifier
def get_lr_clf():
    return LogisticRegression()

# Initialize support vector machine classifier
def get_svm_clf():
    return svm.SVC(C=10,gamma=0.01,kernel='rbf')

# Function for max-min normalization (column-wise)
def max_min_norm(f):
    return (f - f.min(axis=0)) / (f.max(axis=0) - f.min(axis=0))

# Function for standard normalization (column-wise)
def get_std_norm(f):
    return (f - np.average(f, axis = 0)) / np.std(f, axis = 0)

# main function
def main():
    current_working_dir = os.getcwd()
    features = []
    labels = []
    test_features = []
    test_ids = []
    feat = []
    
    # Parse and extract feature from training audio files
    for curr_genre in genre_list:
         print("Current genre: " + curr_genre)
         for curr_file_name in os.listdir(current_working_dir + "/genres/" + curr_genre):
             if curr_file_name.endswith("wav"):
                 feat = []
                 if feature_mode == 1:
                     feat = get_fft_feat("genres/" + curr_genre + "/" + curr_file_name)
                 elif feature_mode == 2:
                     feat = get_mfcc_feat("genres/" + curr_genre + "/" + curr_file_name)
                 else:
                     feat = get_custom_feat("genres/" + curr_genre + "/" + curr_file_name)
                 features.append(feat)
                 labels.append(curr_genre)
    
    # Parse and extract feature from testing audio files
    for curr_file_name in validation_files:
        feat = []
        if feature_mode == 1:
            feat = get_fft_feat("validation/rename/" + curr_file_name + ".wav")
        elif feature_mode == 2:
            feat = get_mfcc_feat("validation/rename/" + curr_file_name + ".wav")
        else:
            feat = get_custom_feat("validation/rename/" + curr_file_name + ".wav")
        test_features.append(feat)
    
    
    features = np.array(features)
    test_features = np.array(test_features)
    
    # Normalize training and testing data (column-wise)
    features = get_std_norm(features)
    test_features = get_std_norm(test_features)
    
    # Normalize training and testing data (row-wise)
    normalize(features, axis=1)
    normalize(test_features, axis=1)
    
    # Separate training and validation dataset randomly for 10-fold cross validation
    features, X_test, labels, y_test = train_test_split(features, labels, test_size=0.10)
    
    # Initilize classifier according to classifier mode
    clf = get_svm_clf()
    
    if classifier_mode == 1:
        clf = get_lr_clf()
    
    # Train our classifier
    model = clf.fit(features, labels)
    
    # Make predictions
    preds = clf.predict(test_features)
    print(preds)
    
    # Write prediction to output file
    fid = open("results.txt", "w")
    fid.write("id,class\n")
    i = 0;
    for curr_id in test_ids:
        fid.write(str(curr_id[:-3]) + "au," + preds[i] + "\n")
        i = i + 1
    fid.close()
    
    # Make prediction for 10-fold validation
    preds = clf.predict(X_test)
    
    # Calculate and print 10-fold accuracy
    score = accuracy_score(y_test, preds)
    print("10 fold accuray: ")
    print(score)
    
if __name__ == "__main__":
    main()

