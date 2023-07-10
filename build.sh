#!/bin/sh
rm synapse 

gcc -Wall -Wextra -o xor xor.c -lm

./xor