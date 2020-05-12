import numpy as np
import serial
import time
import pydub
import scipy.io.wavfile
import pygame
import scipy
import wave
import os

#read wav file
#rate,audData=scipy.io.wavfile.read("music/missyou.wav")
#(rate,audData) = wav.read(StringIO.StringIO("music/missyou.wav"))

def load_data(path):
    data = []
    label_index = np.array([], dtype=int)
    label_count = 0
    wav_files_count = 0

    for root, dirs, files in os.walk(path):
        # get all wav files in current dir 
        wav_files = [file for file in files if file.endswith('.wav')]
        data_same_person = []
        # extract logfbank features from wav file
        for wav_file in wav_files:
            (rate, sig) = wav.read(root + "/" + wav_file)
            fbank_beats = logfbank(sig, rate, nfilt=40)
            # save logfbank features into same person array
            data_same_person.append(fbank_beats)

        # save all data of same person into the data array
        # the length of data array is number of speakers
        if wav_files:
            wav_files_count += len(wav_files)
            data.append(data_same_person)

    # return data, np.arange(len(data))
    return data 
datala = load_data("/music/")
print(datala)
waitTime = 0.1

# generate the waveform table
signalLength = 1024
t = np.linspace(0, 2*np.pi, signalLength)
#signalTable = (np.sin(t) + 1.0) / 2.0 * ((1<<16) - 1)

# output formatter
formatter = lambda x: "%.3f" % x

# send the waveform table to K66F
serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)
print("Sending signal ...")
print("It may take about %d seconds ..." % (int(signalLength * waitTime)))
i=0
for data in datala:
  s.write(bytes(formatter(data), 'UTF-8'))
  i = i + 1
  time.sleep(waitTime)
  if i == 50 :
     break
s.close()
print("Signal sended")