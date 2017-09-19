import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from os import path, listdir
import pickle
import re
import sqlite3 as sqlite

dirpath = path.dirname(path.abspath('__file__'))
orig_list = listdir(
    path.join(dirpath, 'Thistles/jer_2016jan13n15/barley_10m/orig/'))

# build one connection to db file
con = sqlite.connect('barley_10m.db')
c = con.cursor()
print('creating table...')
c.execute('create table img(im_id, rgb, hsv)')
print('table has been created')

for orig_name in orig_list:
    img_id = re.split('\.|_', orig_name)[1]
    img_rgb = plt.imread(
        path.join('Thistles/jer_2016jan13n15/barley_10m/orig', orig_name))
    img_hsv = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2HSV)
    
    c.execute(
        'insert into img(im_id, rgb, hsv) values(?, ?, ?)',
        (img_id, pickle.dumps(img_rgb), pickle.dumps(img_hsv))
    )
    con.commit()
    print('Inserting img: {}'.format(img_id))
    

con.close()

