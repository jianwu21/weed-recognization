from __future__ import division

import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import (
    listdir,
    path,
)
import pickle
import re
import sqlite3 as sqlite

def corlor_normalization(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mean_s = np.mean(img_hsv[:,:,1])
    mean_v = np.mean(img_hsv[:,:,2])

    std_s = np.std(img_hsv[:,:,1])
    std_v = np.std(img_hsv[:,:,2])

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img_hsv[x, y, 1] = (img[x, y, 1]-mean_s)/std_s
            img_hsv[x, y, 2] = (img[x, y, 2]-mean_v)/std_v

    return img_hsv


def cut_patches(img):
    '''
    The return value is a dictionary, the key is (x, y) in
    orgin img, the values is np.array.
    '''
    l, w, d = img.shape

    num_l = l/100
    num_w = w/100

    map_imgs = {}

    for i in range(num_l):
        for j in range(num_w):
            map_imgs.update({(i, j): img[100*i:100*(i+1), 100*j:100*(j+1)]})

    return map_imgs
