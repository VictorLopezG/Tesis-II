import pywt 
import numpy as np

def DWT(polar_array):
    coeficientes=pywt.dwt2(polar_array, 'db9')
    LL,(LH,HL,HH)=coeficientes
    temp=np.hstack([LL])
    mask=np.ones(temp.shape,np.int8)
    return temp,mask