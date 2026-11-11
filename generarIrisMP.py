from IrisGen.LRII import PCAGen
from IrisGen.SVDII import SVDGen
from IrisGen.mi_pcr import mi_pcrGen
from os import path,listdir
from cv2 import imwrite, imread
from Segmentacion.normalize import rubber_sheet_inverse
from multiprocessing import Process
import numpy as np

basePath="C:\\Users\\wings\\Desktop\\CASIA\\TEST"
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
    imgs=[]
    ids=listdir(dir)
    for i in ids:
        ims=listdir(f"{dir}/{i}/")
        imgs.append(imread(f"{dir}/{i}/{ims[0]}",0))
    return imgs

def generarLote(cod:str,combinaciones:list,imgs:list):
    #print(ini,fin,combinaciones[ini:fin])
    ##print(ini, fin)
    for (a,b,c,d,e) in combinaciones:
        ims=[imgs[a],imgs[b],imgs[c],imgs[d],imgs[e]]
        match cod:
            case 'PCA':
                new=PCAGen(imgs=ims)
            case 'MI_PCR':
                new=mi_pcrGen(imgs=ims)
            case 'SVD':
                new=SVDGen(imgs=ims)
        circ= rubber_sheet_inverse(np.asarray(new),(290,320),50,200)
        name=f"{a}_{b}_{c}_{d}_{e}.jpg"
        imwrite(f"imputedImgs/{cod}/{name}",new)
        imwrite(f"imputedImgs/{cod}/C{name}",circ)
        

if __name__=="__main__":
    combinaciones=[]
    imgs=loadImgs(basePath)
    try:
        print("Cargando imagenes de base")
        f=open("validas.txt","r")
        n_threads=4
        cant=108
        k=0
        cod='MI_PCR'
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
           hilos.append(Process(target=generarLote,args=(cod,combinaciones[i*(total//n_threads):(i+1)*(total//n_threads)],imgs)))
        for h in hilos: h.start()
        for h in hilos: h.join()
    except Exception as e:
        print(e)