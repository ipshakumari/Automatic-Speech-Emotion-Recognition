
import joblib
import librosa
import soundfile
import os, glob, pickle
import numpy as np
#import speech_recognition as sr
import sounddevice as sd
from ownplay import timer, record_audio
from movierecm import getmovie
from scipy.io.wavfile import write
from playsound import playsound
import time
import os
import webbrowser
import requests as HTTP

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


print('##########Load the model######################')
model = joblib.load('mlp')
nm = ''
fl = input('Give Yes for selecting new audio No for old Actor audio[y/n]')

if fl.lower() == 'y':
    filename ="new_record.wav"
    record_audio(filename)

    listen = input("Do you want to listen the recorded audio? [y/n]")

    if listen.lower() == "y":
        os.system("aplay " + filename)
        print('played the sound')
    nm = filename

else:

    ac =input('Actor:')
    getfl = []
    for x in glob.glob('/home/ipsha/PROJECT/speech-emotion-recognition-ravdess-data/Actor_'+ac+'/*.wav'):
        getfl.append(os.path.basename(x))

    print(getfl)
    flnm = input('Select Audio File name with extension:')
    nm = '/home/ipsha/PROJECT/speech-emotion-recognition-ravdess-data/Actor_'+ac+'/'+flnm

    listen = input("Do you want to listen the recorded audio? [y/n]")

    if listen.lower() == "y":
        os.system("aplay " + nm)
        print('played the sound')

x_i = extract_feature(nm, mfcc=True, chroma=True, mel=True)
#print(x_i.shape)
print('It seems you are: ')
print(model.predict([x_i])[0])
print('Here are few movie recommendations for you.')
getmovie(model.predict([x_i])[0])


