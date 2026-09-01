from Codificacion import dct, dwt, pca, gabor
from Segmentacion.normalize import normalize
from Segmentacion.segment import segment
from cv2 import createCLAHE, imshow,waitKey

clahe=createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def process_iris(img,cod:str):
    img=clahe.apply(img)
    iris, pupila, noise=segment(img)
    polar_array,noise_array=normalize(noise, iris[1], iris[0], iris[2], pupila[1], pupila[0], pupila[2], 64, 512)    
    #imshow("polar",polar_array)
    #waitKey()
    match cod:
        case "dct":
            temp, mask=dct.extractFeature_dct(polar_array)
        case "dwt":
            temp,mask=dwt.DWT(polar_array)
        case "gabor":
            temp, mask=gabor.extractFeature_gabor(polar_array,noise_array)
        case "pca":
            temp,mask=pca.prCoAn(polar_array)
    return temp, mask
