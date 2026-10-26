from IrisGen.LRII import PCAGen
from IrisGen.SVDII import SVDGen
from IrisGen.mi_pcr import mi_pcrGen
from os import path
from cv2 import imwrite
from Segmentacion.normalize import rubber_sheet_inverse

basePath="C:\\Users\\wings\\Desktop\\CASIA\\TEST"

def getPath(id:str):
    if int(id)>9:
        return path.join(basePath,id,f"0{id}_1_3.jpg")    
    return path.join(basePath,id,f"00{id}_1_3.jpg")

try:
    i=0
    f=open("Combinatoria.txt")
    for line in f:
        a=line[:2]
        if int(a)<10:
            a=line[1:2]
        b=line[2:4]
        if int(b)<10:
            b=line[3:4]
        c=line[4:6]
        if int(c)<10:
            c=line[5:6]
        d=line[6:8]
        if int(d)<10:
            d=line[7:8]
        e=line[8:10]
        if int(e)<10:
            e=line[9:10]
        paths=[getPath(a),getPath(b),getPath(c),getPath(d),getPath(e)]
        newIm=SVDGen(paths)
        imwrite(f"imputedImgs\\SVD\\{a}_{b}_{c}_{d}_{e}.jpg",newIm)
        circ=rubber_sheet_inverse(newIm,(290,320),50,200)
        imwrite(f"imputedImgs\\SVD\\C{a}_{b}_{c}_{d}_{e}.jpg",circ)
        i=i+1
        print(f"{i}/30")
except Exception as e:
    print(e)