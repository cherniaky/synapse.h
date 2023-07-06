#!/bin/sh
rm synapse 

gcc -Wall -Wextra -o synapse synapse.c -lm

./synapse