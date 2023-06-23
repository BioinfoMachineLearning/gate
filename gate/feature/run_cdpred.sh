#!/bin/sh
cd /home/bml_casp15/tools/CDPred/

source /home/bml_casp15/tools/CDPred/env/CDPred_virenv/bin/activate

python lib/Model_predict.py -n $1 -p $2 -a $3 -m $4 -o $5
