# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 23:50:33 2019

@author: DongXiaoning
"""
import numpy as np
from sklearn.cluster import KMeans
import PIL.Image as image
import os.path
import scipy.cluster.vq
from scipy.misc import imresize
import pylab



FILE_NAME = 'photo.jpg'

def load_image(file_path):
    with open(file_path,'rb') as rf:
        img = np.array(image.open(rf))
    return img

# square = 2, height_steps = 160
# square = 4, height_steps = 80

def cluster_pix_square(img,k,height_steps):
    curr = os.getcwd()
    file_path = os.path.join(curr,FILE_NAME)
    #img = load_image(file_path)
    img = np.array(image.open(file_path))
    height = img.shape[0]
    width = img.shape[1]
    square = height / height_steps
    width_steps = width / square
    square = int(square)
    width_steps = int(width_steps)
    pixels = []
    for i in range(height_steps):
        for j in range(width_steps):
            R = np.mean(img[i*square:(i+1)*square,j*square:(j+1)*square,0])
            G = np.mean(img[i*square:(i+1)*square,j*square:(j+1)*square,1])
            B = np.mean(img[i*square:(i+1)*square,j*square:(j+1)*square,2])
            pixels.append([R,G,B])
    pixels = np.array(pixels,'f')
    centroids, variance = scipy.cluster.vq.kmeans(pixels, k)
    code, distance = scipy.cluster.vq.vq(pixels, centroids)
    codeim = code.reshape(height_steps, width_steps)
    codeim = imresize(codeim, img.shape[:2], 'nearest')
    return codeim

def main():
    infile = 'photo.jpg'
    im = np.array(image.open(infile))

    m_k = 15

    pylab.figure()
    pylab.subplot(131)
    pylab.title('source')
    pylab.imshow(im)

    codeim= cluster_pix_square(infile, m_k,160)
    pylab.subplot(132)
    pylab.title('Steps = 160, K = '+str(m_k));
    pylab.imshow(codeim)

    codeim= cluster_pix_square(infile, 5, 160)
    pylab.subplot(133)
    pylab.title('Steps = 160, K = 5');
    pylab.imshow(codeim)

    pylab.show()
    return


if __name__=='__main__':
    main()