from Codificacion.process import process_iris
from cv2 import imread, imshow, waitKey, imwrite
from os import path, listdir
from Codificacion.register import register
from Codificacion.process import process_iris
from Codificacion.metrics import euclideanDistance, hammingDistance
from scipy.io import loadmat

codes=['gabor','dct','dwt','pca']
thresholds={'pca':0.0028,'dct':0.03,'gabor':0.38,'dwt':0.0008}

if __name__=='__main__':

    imgs=listdir("Images/")

    for COD in codes:
        dist=euclideanDistance
        if COD=='gabor':
            dist=hammingDistance
        resgistered=listdir(path.join("Templates",COD))

        #registrar todo

        for im in imgs:
        #codificar imagenes
            if im[:-4]+'.mat'not in resgistered:
                register(path.join('Images',im),COD,im[:-4])

        #cargar plantillas
        print("Leyendo plantillas...")

        entrys=listdir(path.join("Templates",COD))

        temps=[]

        for temp in entrys:
            template=loadmat(path.join("Templates",COD,temp))
            temps.append([template['template'],''.join(template['id'])])
        #buscar dentro de la bd
        results=[]
        print(f"probando codificacion: {COD}")
        for im in imgs:
            print(im)
            img=imread('Images/'+im,0)
            for temp in temps:
                enc,mask=process_iris(img,COD)
                d=dist(enc,temp[0])
                if d<thresholds[COD]:
                    results.append([d,im,temp[1]])
                    break

        print(f"FN:{len(results)-len(imgs)}")
        tp=0
        for res in results:
            if res[1][:-4]==res[2]:
                tp=tp+1
            else:
                print(res)
        print(f"TP:{tp/len(imgs)}")
