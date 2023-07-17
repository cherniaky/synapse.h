#!/bin/sh
# exec=dump_nn
# exec=xor
exec=gym
# exec=xor.gen

rm $exec 

gcc -o $exec $exec.c -Wall -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

./$exec xor.arch xor.mat
# ./$exec