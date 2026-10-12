import os.path
from cv2 import imread, imshow, waitKey
from scipy.io import savemat
from Codificacion.process import process_iris
from sklearn.cluster import KMeans,BisectingKMeans,AgglomerativeClustering
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from sklearn.metrics import accuracy_score,precision_score,recall_score
from time import time_ns,time
import numpy as np

thresholds={'pca':0.0028,'dct':0.03,'gabor':0.38,'dwt':0.0008}

def calcCentroid(cluster:list):
    """Calcular el centroide de un cluster"""
    suma=np.zeros(cluster[0].shape)
    for entry in cluster:
        suma+=entry/len(cluster)
    return suma


def register(im:str,cod:str,id:str):
    img = imread(im, 0)

    temp,mask=process_iris(img,cod)

    out_file = os.path.join(f'Templates/{cod}', f"{os.path.basename(im)[0:-4]}.mat")
    savemat(out_file, mdict={'id': id, 'template': temp, 'mask': mask})

def searchCents(im:str,cod:str,clusters:KMeans|BisectingKMeans,table:list, metric):
    """Busqueda en clusteres formados con algoritmos basados en centroides
        im: ruta a la imagen
        cod: codificacion que se uilizo en los clusteres
        clusters: clusteres formados
        table: tabla en que se relaciona cada plantilla con su respectivo cluster e id
        metric: funcion de distancia para las comparaciones
    """
    ini=time_ns()
    img=imread(im,0)
    if img is None :return 'Imagen no valida',time()-ini
    #codificar
    temp,mask=process_iris(img,cod)
    
    #identificar cluster en que deberia encontrarse
    #dist=0
    idx=clusters.predict([temp.flatten()])
    '''
    for ind in range(len(clusters.cluster_centers_)):
        d=metric(clusters.cluster_centers_[ind],temp)
        ##print(d)
        if d<dist or ind-1<0:
            idx=ind
            dist=d
    '''
    #comparar con el cluster
    #print(idx)
    for ind in range(len(table)):
        if table[ind][0]==idx:
            d=metric(temp,table[ind][2])
            ##print(d)
            if d<=thresholds[cod]:
                return table[ind][1],os.path.basename(im)[0:-4],d,time()-ini

    return '-1',os.path.basename(im)[0:-4],-1,time()-ini

def searchTree(im:str,cod:str,clusters:AgglomerativeClustering,table:list, metric):
    """Busqueda en clusteres formados con algoritmos basados en centroides
        im: ruta a la imagen
        cod: codificacion que se uilizo en los clusteres
        clusters: clusteres formados
        table: tabla en que se relaciona cada plantilla con su respectivo cluster e id
        metric: funcion de distancia para las comparaciones
    """
    ini=time()
    img=imread(im,0)
    if img is None :return 'Imagen no valida',time()-ini
    #codificar
    temp,mask=process_iris(img,cod)
    
    #calcular centriodes
    dist=0
    idx=0
    for i in range(clusters.n_clusters_):
        cluster=[]
        for entry in table:
            if entry[0]==i:
                cluster.append(entry[2])
        cent=calcCentroid(cluster)
        d=metric(cent,temp)
        #print(d)
        if d<dist or i-1<0:
            idx=i
            dist=d   

    print(idx) 

    for ind in range(len(table)):
        if table[ind][0]==idx:
            d=metric(temp,table[ind][2])
            #print(d)
            if d<=thresholds[cod]:
                return table[ind][1],os.path.basename(im)[0:-4],d,time()-ini

    return '-1',os.path.basename(im)[0:-4],-1,time()-ini


def searchSCAN(im:str,cod:str,db:HDBSCAN,table:list,metric):
    ini=time()
    img=imread(im,0)
    if img is None :return 'Imagen no valida',time()-ini
    #codificar
    temp,mask=process_iris(img,cod)
    dist=0
    idx=0
    for ind in range(len(db.centroids_)):
        d=metric(db.centroids_[ind],temp)
        if d<dist or ind-1<0:
            idx=ind
            dist=d
    print(idx)
    #comparar con el cluster
    for ind in range(len(table)):
        if table[ind][0]==idx:
            d=metric(temp,table[ind][2])
            #print(d)
            if d<=thresholds[cod]:
                return table[ind][1],os.path.basename(im)[0:-4],d,time()-ini
    
    #Buscar dentro de los outliers si no se encuentra en los principales
    for ind in range(len(table)):
        if table[ind][0]==-1:
            d=metric(temp,table[ind][2])
            #print(d)
            if d<=thresholds[cod]:
                return table[ind][1],os.path.basename(im)[0:-4],d,time()-ini
        else:
            break

    return '-1',os.path.basename(im)[0:-4],-1,time()-ini