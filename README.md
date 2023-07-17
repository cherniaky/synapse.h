# synapse.h
Simple std-style library for training Neural Networks
## Installation
Start by:
docker-compose up -d
docer-compose exec synapse bash

To exit bash shell type "exit"
## gym.c
Example compile options:
```
#!/bin/sh
exec=gym

gcc -o $exec $exec.c -Wall -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

./$exec xor.arch xor.mat
```
It requires arch file and file with training data.
As a result you will see graphical window with cost function plot and representation of neural network that updates its weights and biases on each frame.
![gym example]([http://url/to/img.png](https://raw.githubusercontent.com/CherniakYura/synapse.h/main/gym.jpg)https://raw.githubusercontent.com/CherniakYura/synapse.h/main/gym.jpg)
