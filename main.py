#coding:utf-8
from __future__ import division  
import wave
import struct
from array import *
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
print Fs
sp=spec.specgram(y,8000,Fs,np.hanning(8000),8000)
tsb=1/Fs

f=np.fft.fftfreq(8000,tsb)
print f
data=abs(sp[:,0:(sp[100].size//2)])
dff=np.diff(data,n=2)
dfff=np.diff(data,n=1)
for i in range(np.shape(data)[0]):
	for j in range(500):
		if(dff[i][j+1]<0 and abs((dfff[i][j+1]+dfff[i][j])//2)<3000):
			print i
			print f[j]
			print dff[i][j+1]
			print (dfff[i][j+1]+dfff[i][j])//2
			print data[i][j]
print dff[100].size
print data[100].size
print sp[100].size//2
p1=subplot(211)
p2=subplot(212)
p3=subplot(221)
p1.plot(f[:(f.size//2)],abs(data[100]))
p2.plot(f[:(f.size//2)-2],dff[100])
p3.plot(f[:(f.size//2)-1],dfff[100])
show()


#plot(y)
#show()

