#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 19:01:53 2021

@author: ipsha
"""
import joblib
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))#40
            #if f == 0:
                #print(mfccs)
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))#12
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))#128
    return result
#Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
  
}


#Emotions to observe
observed_emotions=['sad', 'happy', 'angry']#['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
#
#Load the data and extract features for each sound file

def load_data(test_size=0.2):
    x,y=[],[]
    #fl = 1
    for file in glob.glob("/home/ipsha/PROJECT/speech-emotion-recognition-ravdess-data/Actor_*/*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        #if fl == 0:
            #print(file,emotion,feature)
            #fl = fl+1
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

print('############Load dataset and extract features############')
#Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.10)
#Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))
#Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')
print('############Train MLP model for our training data############')
#Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier( hidden_layer_sizes=(200,), learning_rate='adaptive', max_iter=400)#alpha=0.01, batch_size=256, epsilon=1e-08,
#Train the model
model.fit(x_train,y_train)
#Predict for the test set

print('############Predict data using our model############')
#print(x_test)
y_pred=model.predict(x_test)
#print(y_pred)
#Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))
print("######################Save the model#######")
joblib.dump(model,'mlp')
"""
ac =input('Actor:')
getfl = []
for x in glob.glob('/home/ipsha/PROJECT/speech-emotion-recognition-ravdess-data/Actor_'+ac+'/*.wav'):
    getfl.append(os.path.basename(x))

print(getfl)
flnm = input('Select Audio File name with extension:')
nm = '/home/ipsha/PROJECT/speech-emotion-recognition-ravdess-data/Actor_'+ac+'/'+flnm
x_i = extract_feature(nm, mfcc=True, chroma=True, mel=True)
print(model.predict([x_i])[0])
"""
