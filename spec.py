#coding:UTF-8
import numpy as np
from array import *
def specgram(data,niff,fs,window,diff):
	res=np.array(np.fft.fft((data[0:niff])*window))
	e=0
	for i in range((data.size-niff)//diff) :
		res=np.vstack((res,np.fft.fft((data[i*diff:i*diff+niff])*window)))
		print i
	return res
