import sounddevice as sd
from scipy.io.wavfile import write
from playsound import playsound
import time
import os

def timer(duration):
    while duration: 
        mins, secs = divmod(duration, 60) 
        timer = f"5 secs:{secs} seconds Left"
        print(timer, end=" \r") 
        time.sleep(1) 
        duration -= 1

def record_audio(filename):
    
    #frequency
    fs=44100  #frames per seconds  
    duration = 5  # seconds in integer
    
    print("Recording..........")

    #start recording 
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)

    timer(duration)    #call timer function
    sd.wait()

    #write the data in filename and save it
    write(filename, fs, myrecording)

#filename ="new_record.wav"
"""
record_audio(filename)

listen = input("Do you want to listen the recorded audio? [y/n]")

if listen.lower() == "y":
    os.system("aplay " + filename)
    print('played the sound')
"""
