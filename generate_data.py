import cv2
import matplotlib.pyplot as plt
from os import (
    listdir,
    path,
)
import pickle
import sqlite3 as sqlite


def main():
    dirpath = path.dirname(path.abspath('__file__'))
    img_list = listdir(
        path.join(
            dirpath,
            'Thistles/jer_2016jan13n15/barley_30m/output_barley_30m/'))

    # build one connection to db file
    con = sqlite.connect('barley_30m.db')
    c = con.cursor()
    print('creating table...')
    c.execute(
        '''
        create table img(
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
        if not 'surround' in img_name:
            rgb = plt.imread(
                path.join(
                    'Thistles/jer_2016jan13n15/barley_30m/output_barley_30m/',
                    img_name
                )).astype('float32')
            hsv = cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV)
            if 'uc' in img_name:
                label = 1
            else:
                label = 0

            c.execute(
                'insert into img(rgb, hsv, label) values(?, ?, ?)',
                (pickle.dumps(rgb), pickle.dumps(hsv), label)
            )
            print('Inserting img: {}'.format(img_name))
        con.commit()

    con.close()


if __name__ == '__main__':
    main()
