Weed Recognization
==================

This is the free topic for 2017-2018 block 1. You can see all [experiments](https://github.com/JaggerWu/weed-recognization/blob/master/weed%20recognization.ipynb) run by me in a jupyter notebook file.

-----

Database description
--------------------

The database include two kinds of crops,barley and wheat.For each group,we got theimages from 5 attitutes,10m,20m,30,50m.   
<img src="./example_Img/IMG_7347.JPG" title="Barley 10m" width="200" height="200"/>
<img src="./example_Img/IMG_7317.JPG" title="Barley 20m" width="200" height="200"/> 
<img src="./example_Img/IMG_7278.JPG" title="Barley 30m" width="200" height="200"/>
<img src="./example_Img/IMG_7291.JPG" title="Barley 50m" width="200" height="200"/>

For each image, we divide it to several small pictures.Then we identify these pictures as weed one or crop one mamully.Atfer we extend the pictures to one surround picture for training.   
<img src="./example_Img/IMG_7347_fn_07_13_ttc_uw.png" width="200" height="200">
<img src="./example_Img/IMG_7347_fn_07_13_ttc_uw_surround.png" width="200" height="200">

Before start
------------
### Data Generation

All imges are in ``.JPG`` format. It will take sometime for `python` to load it. 
There is one script for data generation. All image will be stored in one `.db` file with both of RGB and HSV space. Then it will be convinient and fast for us to extract the matrix of images, which can be helpful to improve experiment speed.

```shell
python -m generate_data.py
```

### python for Linux

There are some problems with python in Linux environment. So please remember to set virtual environment or virtual machine in Linux.I use [virtualenv](https://virtualenv.pypa.io/en/stable/)
```shell
virtualenv env
source env/bin/activate
```
For preprocess our images, I choose [OpenCV](http://opencv.org/)


### [CNN](http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/)
CNN is so hot now.It made something easier.Instead of old school image processing method,cnn seems a better for us to do area recognition.

Preprocess for images
---------------------

- Color normalization
- Images normalization
- Remove the edge with value 0

### Color normalization

Since our pictures are taken from different weather and different angle. So the light may have significant influence in the images for training and test. Converting it from RGB space to HSV space is a good idea for the origin images to be normalized. Because [HSV/HSL](https://en.wikipedia.org/wiki/HSL_and_HSV)(Hue, Saturation, Value/Lightness) space can express the relationship with light directly.   

Here, the effect is concentrated both in the saturation component and the value component, whereas the hue component is hardly affected.We choose to correct both saturation $S$ and value $V$ using a linerar fit to each image, i.e:

$$ S(x, y) \frac{\overline(S)}{a_s+b_sx+c_sy}S(x, y)$$

### Image normalization

Make all images including traing set and testing set be in the same scale

### Remove the edges with value 0

It is necessary to remove these small parts. I think the pixel with value 0 may have bad influence on classifier training. If we remove the pixels which are 0, the accurance will increase. Not sure now. Will experiment!!!

The **issue** is if we should removw these ones for just use these pixels which are not value of 0 for training.

<img src="./example_Img/IMG_7347_fn_07_13_ttc_uw_surround.png" width="200" height="200">

Data Training
-------------

Single layer CNN????? Will come later.

