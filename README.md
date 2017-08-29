Weed Recognization
==================

This is the free topic for 2017-2018 block 1.

-----

Database description
--------------------

The database include two kinds of crops,barley and wheat.For each group,we got the images from 5 attitutes,10m,20m,30,50m.   
<img src="./example_Img/IMG_7347.JPG" title="Barley 10m" width="200" height="200"/>
<img src="./example_Img/IMG_7317.JPG" title="Barley 20m" width="200" height="200"/> 
<img src="./example_Img/IMG_7278.JPG" title="Barley 30m" width="200" height="200"/>
<img src="./example_Img/IMG_7291.JPG" title="Barley 50m" width="200" height="200"/>

For each image, we divide it to several small pictures.Then we identify these pictures as weed one or crop one mamully.Atfer we extend the pictures to one surround picture for training.   
<img src="./example_Img/IMG_7347_fn_07_13_ttc_uw.png" width="200" height="200">
<img src="./example_Img/IMG_7347_fn_07_13_ttc_uw_surround.png" width="200" height="200">

Python for Linux
----------------

There are some problems with python in Linux environment. So please remember to set virtual environment or virtual machine in Linux.I use [virtualenv](https://virtualenv.pypa.io/en/stable/)
```shell
virtualenv env
source env/bin/activate
```

[CNN](http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/)
--------------------------------------------------------------------------------------------
CNN is so hot now.It made something easier.Instead of old school image processing method,cnn seems a better for us to do area recognition.
