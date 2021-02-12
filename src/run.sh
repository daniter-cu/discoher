#! /bin/bash

python training.py --contrasts=16 --emsize 768 --nhid 1536 --bptt 256 --batch_size 16 \
--nlayers 2 --log-interval 50 --lr 1 --cuda --save_interval 100
