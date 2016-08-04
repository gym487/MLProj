#coding:UTF-8
import numpy as np
from array import *
def specgram(data,niff,fs,window,diff):
	
	res=np.zeros(((data.size-niff)//diff,niff))
	for i in range((data.size-niff)//diff) :
		res[i]=abs(np.fft.fft((data[i*diff:i*diff+niff])*window))
		print i
	return res

