import os.path
from cv2 import imread, imshow, waitKey
from scipy.io import savemat
from Codificacion.process import process_iris
from sklearn.cluster import KMeans,BisectingKMeans,DBSCAN,AgglomerativeClustering
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from sklearn.metrics import accuracy_score,precision_score,recall_score
import numpy as np

thresholds={'dct':1300,'gabor':0.38,'dwt':170,'pca':170}

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
    img=imread(im,0)
    #codificar
    temp,mask=process_iris(img,cod)

    #identificar cluster en que deberia encontrarse
    dist=0
    idx=0
    for ind in range(len(clusters.cluster_centers_)):
        d=metric(clusters.cluster_centers_[ind],temp)
        if d<dist or ind-1<0:
            idx=ind
            dist=d

    #comparar con el cluster
    dists=[]
    for ind in range(len(table)):
        if table[ind][0]==idx:
            dists.append([metric(temp,table[ind][2]),table[ind][1]])

    dists.sort()

    if dists[0][0]>thresholds[cod]:
            return ['El individuo no se encuentra registrado', -1]

    return dists[0][1],dists[0][0]

def searchTree(im:str,cod:str,clusters:AgglomerativeClustering,table:list, metric):
    """Busqueda en clusteres formados con algoritmos basados en centroides
        im: ruta a la imagen
        cod: codificacion que se uilizo en los clusteres
        clusters: clusteres formados
        table: tabla en que se relaciona cada plantilla con su respectivo cluster e id
        metric: funcion de distancia para las comparaciones
    """
    img=imread(im,0)
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
        if d<dist or i-1<0:
            idx=i
            dist=d    

    dists=[]

    for ind in range(len(table)):
        if table[ind][0]==idx:
            dists.append([metric(temp,table[ind][2]),table[ind][1]])
    
    dists.sort()

    if dists[0][0]>thresholds[cod]:
            return ['El individuo no se encuentra registrado', -1]

    return dists[0][1],dists[0][0]


def searchSCAN(im:str,cod:str,db:HDBSCAN,table:list,metric):
    img=imread(im,0)
    #codificar
    temp,mask=process_iris(img,cod)

    dist=0
    idx=0
    for ind in range(len(db.centroids_)):
        d=metric(db.centroids_[ind],temp)
        if d<dist or ind-1<0:
            idx=ind
            dist=d

    #comparar con el cluster
    dists=[]
    for ind in range(len(table)):
        if table[ind][0]==idx:
            dists.append([metric(temp,table[ind][2]),table[ind][1]])
    dists.sort()

    #Buscar dentro de los outliers si no se encuentra en los principales
    if dists[0][0]>thresholds[cod]:
        dists=[]
        for ind in range(len(table)):
            if table[ind][0]==-1:
                dists.append([metric(temp,table[ind][2]),table[ind][1]])
            else:
                break

    dists.sort()

    if dists[0][0]>thresholds[cod]:
        return ['El individuo no se encuentra registrado', -1]
    
    return dists[0][1],dists[0][0]