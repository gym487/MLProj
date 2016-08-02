#coding:UTF-8
import numpy as np

def specgram(data,niff,fs,window,diff):
	res=np.array([])
	e=0
	for i in range(np.floor((data.size/diff)-niff)) :
		res[i]=np.fft.fft(np.dot(data[i*diff:i*diff+niff],window))
	
	return res

 
		
	
