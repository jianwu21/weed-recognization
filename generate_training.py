from os import path
import sqlite3 as sqlite

def main():
    dirpath = path.dirname(path.abspath('__file__'))
    imglist = listdir(
        path.join(
            dirpath,
            'Thistles/jer_2016jan13n15/barley_10m/')
    )
    
