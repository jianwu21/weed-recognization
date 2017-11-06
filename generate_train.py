import cv2
import matplotlib.pyplot as plt
from os import (
    listdir,
    path,
)
import pickle
import sqlite3 as sqlite
import numpy as np


def main():
    dirpath = path.dirname(path.abspath('__file__'))
    img_list = listdir(
        path.join(
            dirpath,
            'Thistles/jer_2016jan13n15/barley_30m/orig/'))
    la_list = listdir(
        path.join(
            dirpath,
            'Thistles/jer_2016jan13n15/barley_30m/segmentation/'
        )
    )

    # build one connection to db file
    con = sqlite.connect('barley_30m.db')
    c = con.cursor()
    print('creating table...')
    c.execute(
        '''
        create table img_tr(
            id INTEGER PRIMARY KEY AUTOINCREMENT not null,
            rgb,
            hsv,
            label
        );
        '''
    )
    print('table has been created')

    # import the images data
    for img_name in img_list:
        print(img_name)
        for seg in la_list:
            if img_name.split('.')[0] in seg:
                print(seg)
                L = plt.imread(path.join(
                    dirpath,
                    'Thistles/jer_2016jan13n15/barley_30m/segmentation',
                    seg,
                    'segmentation.png',
                )).astype('float32')
        rgb_whole = plt.imread(path.join(
            dirpath,
            'Thistles/jer_2016jan13n15/barley_30m/orig',
            img_name
        )).astype('float32')

        for i in range(30):
            for j in range(40):
                rgb = rgb_whole[100*i:100*(i+1), 100*j:100*(j+1)]
                hsv = cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV)
                l_sum = np.sum(L[100*i:100*(i+1), 100*j:100*(j+1)])
                if l_sum == 0:
                    label = 1
                elif l_sum == 10000:
                    label = 0

                c.execute(
                    'insert into img_tr(rgb, hsv, label) values(?, ?, ?)',
                    (pickle.dumps(rgb), pickle.dumps(hsv), label)
                )
                print('Inserting img: {}{}{}'.format(img_name, i, j))
        con.commit()

    con.close()

if __name__ == '__main__':
    main()
