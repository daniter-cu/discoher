#! /bin/bash

python training.py --contrasts=16 --emsize 768 --nhid 1536 --bptt 256 --batch_size 16 \
--nlayers 2 --log-interval 200 --lr 1 --cuda --save_interval 500 \
--restore ../checkpoints/run1/chkpt-1-3000-model.pt
