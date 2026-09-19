import numpy as np
from scipy.spatial.distance import hamming
def euclideanDistance(v1,v2):
    return np.linalg.norm(v1.flatten()-v2.flatten())

def hammingDistance(v1,v2):
    return hamming(v1.flatten(),v2.flatten())

def accuracy(preds:list, ids:list):
    tp=0
    for pred, id in preds,ids:
        if pred==id:
            tp=tp+1
    return tp/len(preds)

def eval(preds:list, ids:list):
    tp=0
    tn=0
    fp=0
    fn=0
    for pred, id in preds,ids:
        if pred[0]==id[0]:
            tp=tp+1
        if pred[1]==id[1]:
            tn=tn+1
        if pred[1]!=-1 and pred[0]!=id[0]:
            fp=fp+1
        if id[0]!=-1 and pred[1]>-1:
            fn=fn+1
    return tp,tn,fp,fn


def presicion(preds:list, ids:list):
    tp=0
    for pred, id in preds,ids:
            if pred==id:
                tp=tp+1
    return tp/len(preds)

def recall():
    pass

def f1_score(preds:list,ids:list):
    pres=presicion(preds,ids)
    rec=recall()