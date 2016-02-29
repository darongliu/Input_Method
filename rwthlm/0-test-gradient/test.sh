#!/bin/bash

# N.b.: This test will only succeed if the software was compiled with double
# precision.

../rwthlm --vocab vocab.2batches --unk --train train.2batches --batch-size 2 --self-test test-r10-R10-M10-L10
