import numpy as np
import cv2
from scipy.fftpack import dct
from Segmentacion.segment import segment
from Segmentacion.normalize import normalize


def extractFeature_dct(polar_array):
    # Histogram Eq para mejorar contraste y brillo
    polar_int = polar_array * 255
    polar_int = polar_int.astype(int)
    histogram_array = np.bincount(polar_int.flatten(), minlength=256)
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array / num_pixels
    chistogram_array = np.cumsum(histogram_array)
    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)
    img_list = list(polar_int.flatten())

    eq_img_list = [transform_map[p] for p in img_list]
    eq_img_array = np.reshape(np.asarray(eq_img_list), polar_array.shape)

    # CDT to each 8x8 block
    mat_size = eq_img_array.shape
    dct_mat = np.zeros(mat_size)
    for i in np.r_[:mat_size[0]:8]:
        for j in np.r_[:mat_size[1]:8]:
            dct_mat[i:(i + 8), j:(j + 8)] = dct_2d(eq_img_array[i:(i + 8), j:(j + 8)])

    # plantilla biometrica, mantiene solo valor mas discriminante (top-left)
    template = np.zeros((mat_size[0]//8, mat_size[1]//8))
    for i in np.r_[:mat_size[0]:8]:
        for j in np.r_[:mat_size[1]:8]:
            template[i//8, j//8] = dct_mat[i, j]
    mask = np.ones(template.shape,np.int8)
    """cv2.imshow("temps",template)
    cv2.imshow("mask",mask)
    cv2.imshow("norm",dct_mat)
    cv2.waitKey()"""
    return template, mask


def dct_2d(im):
    return dct(dct(im.T,norm='ortho').T,norm='ortho')
