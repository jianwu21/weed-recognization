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

For each image, we divide it to several small pictures(100*100).Then we identify these pictures as weed one or crop one by using classical image processing method. Then, we use the tiny pictures as training data.
<img src="./example_Img/IMG_7347_fn_07_13_ttc_uw.png" width="200" height="200">

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

Here, the effect is concentrated both in the saturation component and the value component, whereas the hue component is hardly affected.We choose to correct both saturation **S** and value **V** using a linear fit to each image.

### Image normalization

Make all images including traing set and testing set be in the same scale


Data Training
-------------

- Training
- Validation
- Test

### Procedure for Training and Validation
Since the file in the original folder include labeled images and surround images. But the training for us is just the labeled RGB images(**100*100**). The idea for that is to store label, RGB channel and HSV channel in one `sqlite` file. Then you can extract them for training anytime. That you can do that by,
```shell
python generate_data.py
```
After doing that, now we can start training. The classical method for model selection is that we train one classifier based on each randomly picked training images group. Then we pick the best one. Now, since the limit for the time and hardware, I just train one classifier. I will update the amount of training group based on one improved CNN after talking with Francois.

### Test
For the test, we want to use classifier in one entire picture. Firstly, the size of one original images is **3000 4000**. So we should divide the testing original picture to **30*40** batches, every batch's size is **100*100**. Then label them depending on selected best model.

CNN structure
-------------

I use the simplest structure for CNN. Since the limit for hardware and time, I just use 50 images for training, but the accuracy is not good. You can see the result from the [notebook](https://github.com/JaggerWu/weed-recognization/blob/master/weed%20recognization.ipynb). The result is not good.

- conv layer 1
- conv layer 2
- full-conected layer 1
- softmax layer

The nest steps
--------------

- Improve the model(add some preprocess for images, like rotation, changes in HSV channel, optimize in iterations).
- Visualization after each convolution layer
- ?????
