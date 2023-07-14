#!/bin/sh
# exec=dump_nn
# exec=xor
exec=gym

rm $exec 

gcc -o $exec $exec.c -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

./$exec