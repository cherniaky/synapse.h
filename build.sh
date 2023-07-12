#!/bin/sh
# exec=dump_nn
exec=xor

rm $exec 

gcc -o $exec $exec.c -lm

./$exec