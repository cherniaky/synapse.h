#!/bin/sh
# exec=dump_nn
# exec=xor
# exec=gym
# exec=xor.gen
# exec=img2nn
# exec=layout
exec=shape

rm build/$exec 

gcc -o build/$exec demos/$exec.c -Wall -Wextra -lraylib -lGL -lm -I. -lpthread -ldl -lrt -lX11

./build/$exec
# ./$exec xor.arch xor.mat
# ./$exec img.arch img.mat
# ./build/$exec mnist/8.png mnist/6.png
