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
