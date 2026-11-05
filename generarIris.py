from IrisGen.LRII import PCAGen
from IrisGen.SVDII import SVDGen
from IrisGen.mi_pcr import mi_pcrGen
from os import path
from cv2 import imwrite
from Segmentacion.normalize import rubber_sheet_inverse

basePath="C:\\Users\\wings\\Desktop\\CASIA\\TEST"

combinaciones=[]

def checkComps(comps:list):
    for comb in combinaciones:
        i=0
        for c in comps:
            if c in comb:
                i=i+1
        if i>1:
            return False
    return True

def getPath(id:str):
    return path.join(basePath,str(int(id)),f"{id}_1_3.jpg")

try:
    i=0
    f=open("Combinatoria.txt")
    for line in f:
        a=line[:3]
        b=line[3:6]
        c=line[6:9]
        d=line[9:12]
        e=line[12:15]
        if not checkComps([a,b,c,d,e]):
            continue
        combinaciones.append([a,b,c,d,e])
        paths=[getPath(a),getPath(b),getPath(c),getPath(d),getPath(e)]
        print(line)
        newIm=mi_pcrGen(paths)
        imwrite(f"imputedImgs\\MI_PCR\\{a}_{b}_{c}_{d}_{e}.jpg",newIm)
        circ=rubber_sheet_inverse(newIm,(290,320),50,200)
        imwrite(f"imputedImgs\\MI_PCR\\C{a}_{b}_{c}_{d}_{e}.jpg",circ)
        i=i+1
        print(f"{i}")
except Exception as e:
    print(e)