import numpy as np
from skimage import io
from skimage.transform import resize
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Carga las imágenes
img1 = io.imread("C:\\Users\\wings\\Desktop\\1.jpg")
img2 = io.imread("C:\\Users\\wings\\Desktop\\2.jpg")
img3 = io.imread("C:\\Users\\wings\\Desktop\\3.jpg")
img4 = io.imread("C:\\Users\\wings\\Desktop\\4.jpg")
img5 = io.imread("C:\\Users\\wings\\Desktop\\5.jpg")

# Convierte las imágenes a escala de grises
imSize=(480,640)
#asegurar que los tamaños sean iguales
img1=resize(img1,imSize,anti_aliasing=True)
img2=resize(img2,imSize,anti_aliasing=True)
img3=resize(img3,imSize,anti_aliasing=True)
img4=resize(img4,imSize,anti_aliasing=True)
img5=resize(img5,imSize,anti_aliasing=True)

# Aplica el SVD a las imágenes
U1, S1, V1 = svd(img1)
U2, S2, V2 = svd(img2)
U3, S3, V3 = svd(img3)
U4, S4, V4 = svd(img4)
U5, S5, V5 = svd(img5)

print(f'U:{U1.shape} S:{S1.shape} V:{V1.shape}')


############## valores propios
####imagen1
valores_propios1 = np.square(np.diag(S1))
indices_no_nulos1 = np.nonzero(valores_propios1)
valores_propios1_no_nulos1 = valores_propios1[indices_no_nulos1]
######imagen2
valores_propios2 = np.square(np.diag(S2))
indices_no_nulos2 = np.nonzero(valores_propios2)
valores_propios2_no_nulos2 = valores_propios2[indices_no_nulos2]
########imagen3
valores_propios3 = np.square(np.diag(S3))
indices_no_nulos3 = np.nonzero(valores_propios3)
valores_propios3_no_nulos3 = valores_propios3[indices_no_nulos3]
##########imagen4
valores_propios4 = np.square(np.diag(S4))
indices_no_nulos4 = np.nonzero(valores_propios4)
valores_propios4_no_nulos4 = valores_propios4[indices_no_nulos4]
#############imagen5
valores_propios5 = np.square(np.diag(S5))
indices_no_nulos5 = np.nonzero(valores_propios5)
valores_propios5_no_nulos5 = valores_propios5[indices_no_nulos5]



# Crea la imagen sintética combinando los componentes principales de las imágenes originales

k = 120 # Número de componentes principales a utilizar
new_S = np.concatenate((S1[:k], S2[:k], S3[:k], S4[:k], S5[:k]))
new_U = np.concatenate((U1[:, :k], U2[:, :k], U3[:, :k], U4[:, :k], U5[:, :k]), axis=1)
new_V = np.concatenate((V1[:k, :], V2[:k, :], V3[:k, :], V4[:k, :], V5[:k, :]))

print(f'U:{new_U.shape} S:{new_S.shape} V:{new_V.shape}')

# Reconstruye la imagen sintética a partir de la nueva descomposición SVD
synthetic_img = np.dot(new_U, np.dot(np.diag(new_S), new_V))
Us,Ss,Vs=svd(synthetic_img)

valores_propioss = np.square(np.diag(Ss))
indices_no_nuloss = np.nonzero(valores_propioss)
valores_propioss_no_nuloss = valores_propioss[indices_no_nuloss]



# Calcula la desviación estándar de la imagen sintética respecto a las imágenes reales
std1 = np.std(synthetic_img - img1)
std2 = np.std(synthetic_img - img2)
std3 = np.std(synthetic_img - img3)
std4 = np.std(synthetic_img - img4)
std5 = np.std(synthetic_img - img5)


# Muestra las imágenes reales y la imagen sintética en un mismo gráfico
fig, axs = plt.subplots(1, 6)
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])
axs[ 0].imshow(img1,cmap='gray')
axs[ 0].set_title('Imagen 1')
axs[ 1].imshow(img2,cmap='gray')
axs[ 1].set_title('Imagen 2')
axs[ 2].imshow(img3,cmap='gray')
axs[ 2].set_title('Imagen 3')
axs[ 3].imshow(img4,cmap='gray')
axs[ 3].set_title('Imagen 4')
axs[ 4].imshow(img5,cmap='gray')
axs[ 4].set_title('Imagen 5')
axs[ 5].imshow(synthetic_img, cmap='gray')
axs[ 5].set_title('Imagen sintética')



# Muestra la desviación estándar de la imagen sintética respecto a las imágenes reales
print(f"Desviación estándar imagen 1 respecto a la imagen sintetica: {std1:.2f}")
print(f"Desviación estándar imagen 2 respecto a la imagen sintetica : {std2:.2f}")
print(f"Desviación estándar imagen 3 respecto a la imagen sintetica : {std3:.2f}")
print(f"Desviación estándar imagen 4 respecto a la imagen sintetica : {std4:.2f}")
print(f"Desviación estándar imagen 5 respecto a la imagen sintetica : {std5:.2f}")
plt.show()


# Calcula la distancia de Hamming entre la imagen sintética y cada imagen
hamming_distance1 = distance.hamming(synthetic_img.flatten(), img1.flatten())
hamming_distance2 = distance.hamming(synthetic_img.flatten(), img2.flatten())
hamming_distance3 = distance.hamming(synthetic_img.flatten(), img3.flatten())
hamming_distance4 = distance.hamming(synthetic_img.flatten(), img4.flatten())
hamming_distance5 = distance.hamming(synthetic_img.flatten(), img5.flatten())

# Muestra las distancias
print(f"Distancia de Hamming entre la imagen sintética y la imagen 1: {hamming_distance1}")
print(f"Distancia de Hamming entre la imagen sintética y la imagen 2: {hamming_distance2}")
print(f"Distancia de Hamming entre la imagen sintética y la imagen 3: {hamming_distance3}")
print(f"Distancia de Hamming entre la imagen sintética y la imagen 4: {hamming_distance4}")
print(f"Distancia de Hamming entre la imagen sintética y la imagen 5: {hamming_distance5}")

euclidean_distance1 = distance.euclidean(synthetic_img.flatten(), img1.flatten())
euclidean_distance2 = distance.euclidean(synthetic_img.flatten(), img2.flatten())
euclidean_distance3 = distance.euclidean(synthetic_img.flatten(), img3.flatten())
euclidean_distance4 = distance.euclidean(synthetic_img.flatten(), img4.flatten())
euclidean_distance5 = distance.euclidean(synthetic_img.flatten(), img5.flatten())

# Muestra las distancias
print(f"Distancia euclidiana entre la imagen sintética y la imagen 1: {euclidean_distance1}")
print(f"Distancia euclidiana entre la imagen sintética y la imagen 2: {euclidean_distance2}")
print(f"Distancia euclidiana entre la imagen sintética y la imagen 3: {euclidean_distance3}")
print(f"Distancia euclidiana entre la imagen sintética y la imagen 4: {euclidean_distance4}")
print(f"Distancia euclidiana entre la imagen sintética y la imagen 5: {euclidean_distance5}")


# Calcula la distancia de coseno entre la imagen sintética y cada imagen
cosine_distance1 = distance.cosine(synthetic_img.flatten(), img1.flatten())
cosine_distance2 = distance.cosine(synthetic_img.flatten(), img2.flatten())
cosine_distance3 = distance.cosine(synthetic_img.flatten(), img3.flatten())
cosine_distance4 = distance.cosine(synthetic_img.flatten(), img4.flatten())
cosine_distance5 = distance.cosine(synthetic_img.flatten(), img5.flatten())

# Muestra las distancias
print(f"Distancia de coseno entre la imagen sintética y la imagen 1: {cosine_distance1}")
print(f"Distancia de coseno entre la imagen sintética y la imagen 2: {cosine_distance2}")
print(f"Distancia de coseno entre la imagen sintética y la imagen 3: {cosine_distance3}")
print(f"Distancia de coseno entre la imagen sintética y la imagen 4: {cosine_distance4}")
print(f"Distancia de coseno entre la imagen sintética y la imagen 5: {cosine_distance5}")
