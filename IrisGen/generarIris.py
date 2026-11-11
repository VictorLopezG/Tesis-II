from IrisGen.LRII import PCAGen
from IrisGen.SVDII import SVDGen
from IrisGen.mi_pcr import mi_pcrGen
from os import path,listdir
from cv2 import imwrite, imread
from Segmentacion.normalize import rubber_sheet_inverse
from threading import Thread
import numpy as np

basePath="C:\\Users\\wings\\Desktop\\CASIA\\TEST"
combinaciones=[]
imgs=[]
'''
def validateComb(comps:list):
    for comb in combinaciones:
        i=0
        for c in comps:
            if c in comb:
                i=i+1
        if i>2:
            return False
    return True

def getPath(id:str):
    return path.join(basePath,str(int(id)),f"{id}_1_3.jpg")
'''
def loadImgs(dir:str):
    ids=listdir(dir)
    for i in ids:
            ims=listdir(f"{dir}/{i}/")
            imgs.append(imread(f"{dir}/{i}/{ims[0]}",0))

def generarLote(ini:int,fin:int,cod:str):
    #print(ini,fin,combinaciones[ini:fin])
    ##print(ini, fin)
    for i in range(ini,fin):
        c=combinaciones[i]
        ims=[imgs[c[0]],imgs[c[1]],imgs[c[2]],imgs[c[3]],imgs[c[4]]]
        match cod:
            case 'PCA':
                new=PCAGen(imgs=ims)
            case 'MI_PCR':
                new=mi_pcrGen(imgs=ims)
            case 'SVD':
                new=SVDGen(imgs=ims)
        circ= rubber_sheet_inverse(np.asarray(new),(290,320),50,200)
        name=f"{combinaciones[i][0]}_{combinaciones[i][1]}_{combinaciones[i][2]}_{combinaciones[i][3]}_{combinaciones[i][4]}.jpg"
        imwrite(f"imputedImgs/{cod}/{name}",new)
        imwrite(f"imputedImgs/{cod}/C{name}",circ)
        

if __name__=="__main__":
    try:
        print("Cargando imagenes de base")
        loadImgs(basePath)
        f=open("validas.txt","r")
        n_threads=4
        cant=108
        k=0
        cod='SVD'
        print("Cargando combinatorias")
        for line in f:
            a=line[:3]
            b=line[3:6]
            c=line[6:9]
            d=line[9:12]
            e=line[12:15]
            combinaciones.append((int(a)-1,int(b)-1,int(c)-1,int(d)-1,int(e)-1))
            k=k+1
            if cant!=None and k>=cant:
                break
        f.close()
        total=len(combinaciones)
        print("Generando imagenes")
        hilos=[]
        for i in range(n_threads):
           hilos.append(Thread(target=generarLote,args=(i*(total//n_threads),(i+1)*(total//4),cod)))
        for h in hilos: h.start()
        for h in hilos: h.join()
    except Exception as e:
        print(e)