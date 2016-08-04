#coding:utf-8
from __future__ import division  
import wave
import struct
from array import *
from scipy import *
from pylab import *
import numpy as np
import spec 
from sklearn import cluster
from sklearn.externals import joblib

from mpl_toolkits.mplot3d import Axes3D
filename = r'./ss.wav'
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
sp=spec.specgram(y,8000,Fs,np.hanning(8000),2000)
tsb=1/Fs

f=np.fft.fftfreq(8000,tsb)
print f
data=sp[:,0:(sp[100].size//2)]
dff=np.diff(data,n=2)
dfff=np.diff(data,n=1)
peaks=np.zeros((data.shape[0],500))
for i in range(np.shape(data)[0]):
	for j in range(3,500):
		if(dff[i][j-1]<-500000 and abs((dfff[i][j-1]+dfff[i][j])//2)<500000 and data[i][j]>np.amax(data[i])/5):#Or / 7 or 14 or else. 
			print i
			print f[j]
			print dff[i][j-1]
			print (dfff[i][j-1]+dfff[i][j])//2
			print data[i][j]
			peaks[i][j]=data[i][j]

print peaks
nodes=np.zeros((peaks.size,6))
nl=np.zeros((peaks.size,2))
l=0;
for i in range(np.shape(peaks)[0]):
	for j in range(peaks[i].size):
		if(peaks[i][j]!=0):
			#nodes[l][0]=data[i][j]
			for k in range(1,6):
				nodes[l][k]=np.amax(data[i,j*k-k:j*k+k])
			nl[l]=[i,j]
			nodes[l][0]=l
			l+=1
				
			
for i in range(nodes.shape[0]):
	if(nodes[i][1]!=0):
		a=nodes[i][1]
		b=nodes[i][0]
		nodes[i]=nodes[i]/nodes[i][1]
		nodes[i][1]=a
		nodes[i][0]=b
print nodes

ax=plt.subplot(111)
clo=np.zeros((nodes.shape[0],3))
clo[:,0]=nodes[:,1]
clo[:,0]=clo[:,0]-np.amin(clo[:,0])
clo[:,0]=clo[:,0]/np.amax(clo[:,0])
clo[:,1]=1-clo[:,0]
clo[:,2]=1-clo[:,0]
print clo
dat=[]
colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

#for i in range(300,12000):
	#if(nodes[i,1]>=10000000):
		#ax.scatter(log(nodes[i,2]),log(nodes[i,3]),log(nodes[i,4]),c=clo[i])
mp = cluster.SpectralClustering(n_clusters=8,
                                eigen_solver='arpack',
                                affinity="nearest_neighbors")
for i in range(nodes.shape[0]):
	if(nodes[i][1]>5000000):#or 10000000 or 5000000 or else
		dat.append(nodes[i])
datb=np.array(dat)
print log(datb[2:])		
print datb.shape
print mp.fit(log(datb[2:]))

print mp.labels_
print log(datb[2:])

#print mp.inertia_
print nl
for i in range(mp.labels_.shape[0]):
	ax.scatter(nl[datb[i,0]//1][0],nl[datb[i,0]//1][1],color=colors[mp.labels_[i]],s=3)
#p1=subplot(211)
#p2=subplot(212)
#p3=subplot(221)
#p1.plot(f[:(f.size//2)],abs(data[100]))
#plot(f[:(f.size//2)-2],dff[100])
#plot(f[:(f.size//2)-1],dfff[100])
#imagesc(f,data)
show()


#plot(y)
#show()

