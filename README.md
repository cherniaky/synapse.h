# synapse.h
Simple std-style library for training Neural Networks

# gym.h
Addition to synapse.h, gives you ability to render display your training results in GUI using raylib

## Prerequisites
You will need to install locally this packages:
1. raylib.h 
2. build-essentials

## Demos
In the demos folder you can find a few demos that uses synapse.h and gym.h

### shape.c
Trains NN that classifies shapes. It trained to distinguish rectangles and circles<br/>
In the right top corner there is drawable canvas, where you can make a shape and NN will try to figure out what shape it is<br/>
Also you can press "Q" or "W" to generate cirle or rectangle on this canvas<br/>
![shape](shape.jpg)

### img2nn.c
Example compile options:
```
#!/bin/sh
exec=gym

gcc -o $exec $exec.c -Wall -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

./$exec xor.arch xor.mat
```
It requires arch file and file with training data.

As a result you will see graphical window with:
1. Cost function plot
2. Representation of neural network that updates its weights and biases on each frame

![gym example](gym.jpg)
## img2mat.c
Example compile options:
```
#!/bin/sh
exec=img2mat

gcc -o $exec $exec.c -O3 -Wall -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

./$exec mnist/training/8/10057.png
```
It requires 8-bit and grayscale img file.

As a result you will see graphical window with:
1. Cost function plot
2. Representation of neural network that updates its weights and biases on each frame
3. Current ability of neural network to upscale given image (in our example we upscale 32x32 image to 64x64)

![img2mat example](img2mat.jpg)
