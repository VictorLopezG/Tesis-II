from Codificacion.process import process_iris
from Segmentacion.normalize import normalize_iris
from Segmentacion.segment import segment
from cv2 import imread, imshow, waitKey, imwrite

mat=imread("Images/4.bmp",0)

imshow("mat",mat)
waitKey()

iris, pupila, noise=segment(mat)

polar=normalize_iris(mat,pupila[:2],pupila[2],iris[:2],iris[2])
imwrite("norm.bmp",polar)
imshow("mat",polar)
waitKey()

temp,mask=process_iris(mat,'dct')

imwrite("dct.jpg",temp*255)
imshow("dct",temp)
waitKey()

temp,mask=process_iris(mat,'pca')

imwrite("pca.jpg",temp*255)
imshow("pca",temp)
waitKey()

temp,mask=process_iris(mat,'dwt')

imwrite("dwt.jpg",temp*255)
imshow("dwt",temp)
waitKey()

temp,mask=process_iris(mat,'gabor')

imwrite("gabor.jpg",temp*255)
imshow("gabor",temp)
waitKey()