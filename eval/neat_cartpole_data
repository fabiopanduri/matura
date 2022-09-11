#!/bin/bash

for ((i = 0 ; i < 100000; i++)); do
    python3 main.py neat -g cartpole -i 200 -s
    cat $i > iteration.txt
    echo $i
    sleep 2
done
