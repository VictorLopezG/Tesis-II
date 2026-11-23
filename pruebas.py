from cv2 import imread,waitKey,imshow, imwrite,resize,circle
from os import listdir
from Segmentacion.normalize import rubber_sheet_inverse,normalize_iris,normalize,circlecoords
from Segmentacion.segment import segment
from Codificacion.process import normalizeIris
path=["CASIA/TEST/1/001_1_3.jpg",'imputedImgs/IrisSint/A_125200_iris_sintetico.jpg','imputedImgs/Circs/IrisSint/CA_125199_iris_sintetico.jpg']
circ=imread(path[0],0)
norm=imread(path[1],0)
#norm,_=normalizeIris(circ)
norm=resize(norm,(512,64))
norm=circle(norm, (130,120),100, (0,255,0), -1)
norm=circle(norm, (380,130),80, (0,255,0), -1)

#norm2,noise_a=normalizeIris(circ)
iris, pupila, noise=segment(circ)
norm3=normalize_iris(noise,(pupila[1],pupila[0]),pupila[2],(iris[1],iris[0]),iris[2])
norm4,noise_arr=normalize(noise, iris[1], iris[0], iris[2], pupila[1], pupila[0], pupila[2], 64, 512)
x,y=circlecoords((iris[1],iris[0]),iris[2]*2.5,circ.shape)
circ[y,x]=255
x,y=circlecoords((pupila[1],pupila[0]),pupila[2],circ.shape)
circ[y,x]=255
imshow("N",norm)
waitKey()
imshow("C",circ)
waitKey()
#imshow("N2",norm2)
#waitKey()
imshow("N3",norm3)
waitKey()
imshow("N4",norm4)
waitKey()
#circ= rubber_sheet_inverse(mat[::-1],(290,320),50,200)
    