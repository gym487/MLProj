#coding:utf-8
import wave
import struct
from scipy import *
from pylab import *
import numpy as np
import spec 


filename = r'./qjj.wav'
wavefile = wave.open(filename, 'rb') # open for writing


nchannels = wavefile.getnchannels()
sample_width = wavefile.getsampwidth()
framerate = wavefile.getframerate()
numframes = wavefile.getnframes()


y = zeros(numframes)


for i in range(numframes):
    val = wavefile.readframes(1)
    left = val[0:2]
#right = val[2:4]
    v = struct.unpack('h', left )[0]
    y[i] = v


Fs = framerate
spec.spec(y,8000,Fs,np.hanning(8000),7000)
#plot(y)
#show()

