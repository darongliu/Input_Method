#!/bin/bash

# The class GradientTest gives us the information that the gradient is
# implemented correctly.
#
# With this test we want to verify whether the accumulation of multiple
# gradients for consecutive mini-batches within a single epoch works correctly
# as well.
#
# At the end, "testaona1" and "testbona1" should be the same, except that
# models were read from different files for the gradient test, and that the
# models differ in the number epochs they were trained.

../rwthlm --no-bias --vocab v --train a1 --dev a2 --learning-rate 0.1 --batch-size 4 --max-epoch 1 --word-wrapping verbatim --no-shuffling tmp/testa-r10-R10-M10-L10
../rwthlm --no-bias --vocab v --train a2 --dev a2 --learning-rate 0.1 --batch-size 4 --max-epoch 2 --word-wrapping verbatim --no-shuffling tmp/testa-r10-R10-M10-L10
../rwthlm --no-bias --vocab v --train b  --dev a2 --learning-rate 0.1 --batch-size 4 --max-epoch 1 --word-wrapping verbatim --no-shuffling tmp/testb-r10-R10-M10-L10

../rwthlm --no-bias --vocab v --train a1 --batch-size 7 --word-wrapping verbatim --self-test tmp/testa-r10-R10-M10-L10 > tmp/testaona1
../rwthlm --no-bias --vocab v --train a1 --batch-size 7 --word-wrapping verbatim --self-test tmp/testb-r10-R10-M10-L10 > tmp/testbona1

diff tmp/test[ab]ona1
rm tmp/test[ab]-r10-R10-M10-L10{,.bk}
rm tmp/test[ab]ona1
