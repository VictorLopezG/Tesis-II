from cv2 import imread,waitKey,imshow, imwrite,resize
from os import listdir
from Segmentacion.normalize import rubber_sheet_inverse
lista=listdir("imputedImgs/IrisSint")
#print(lista)
for img in lista:
    print(img)
    mat=imread("imputedImgs/IrisSint/"+img,0)
    imshow("",mat)
    waitKey()
    mat=resize(mat,(512,64))
    imshow("",mat)
    waitKey()
    circ= rubber_sheet_inverse(mat[::-1],(290,320),50,200)
    #imwrite("imputedImgs/IrisSint/C"+img,circ)