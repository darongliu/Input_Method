#!/bin/bash

# The class GradientTest gives us the information that the gradient is
# implemented correctly.
#
# With this test we want to verify whether the training of several consecutive
# epochs leads to the same result as training epoch by epoch (where the
# training is resumed from file).
#
# N.b.: Even when the data are shuffled, this test will succeed as the exact
# order of the training data can be reproduced as they are sorted each time
# before shuffling.
#
# At the end, "test1ona" and "test2ona" should be the same except that models
# were read from different files for the gradient test.

../rwthlm --vocab v --train a --dev a --learning-rate 0.1 --batch-size 7 --max-epoch  1 --word-wrapping verbatim tmp/test1-r10-R10-M10-L10
../rwthlm --vocab v --train a --dev a --learning-rate 0.1 --batch-size 7 --max-epoch  2 --word-wrapping verbatim tmp/test1-r10-R10-M10-L10
../rwthlm --vocab v --train a --dev a --learning-rate 0.1 --batch-size 7 --max-epoch  3 --word-wrapping verbatim tmp/test1-r10-R10-M10-L10
../rwthlm --vocab v --train a --dev a --learning-rate 0.1 --batch-size 7 --max-epoch  4 --word-wrapping verbatim tmp/test1-r10-R10-M10-L10
../rwthlm --vocab v --train a --dev a --learning-rate 0.1 --batch-size 7 --max-epoch  5 --word-wrapping verbatim tmp/test1-r10-R10-M10-L10
../rwthlm --vocab v --train a --dev a --learning-rate 0.1 --batch-size 7 --max-epoch  6 --word-wrapping verbatim tmp/test1-r10-R10-M10-L10
../rwthlm --vocab v --train a --dev a --learning-rate 0.1 --batch-size 7 --max-epoch  7 --word-wrapping verbatim tmp/test1-r10-R10-M10-L10
../rwthlm --vocab v --train a --dev a --learning-rate 0.1 --batch-size 7 --max-epoch  8 --word-wrapping verbatim tmp/test1-r10-R10-M10-L10
../rwthlm --vocab v --train a --dev a --learning-rate 0.1 --batch-size 7 --max-epoch  9 --word-wrapping verbatim tmp/test1-r10-R10-M10-L10
../rwthlm --vocab v --train a --dev a --learning-rate 0.1 --batch-size 7 --max-epoch 10 --word-wrapping verbatim tmp/test1-r10-R10-M10-L10
../rwthlm --vocab v --train a --dev a --learning-rate 0.1 --batch-size 7 --max-epoch 10 --word-wrapping verbatim tmp/test2-r10-R10-M10-L10

../rwthlm --vocab v --train a --word-wrapping verbatim --batch-size 7 --self-test tmp/test1-r10-R10-M10-L10 > tmp/test1ona
../rwthlm --vocab v --train a --word-wrapping verbatim --batch-size 7 --self-test tmp/test2-r10-R10-M10-L10 > tmp/test2ona

diff tmp/test[12]ona

rm tmp/test1-r10-R10-M10-L10
rm tmp/test2-r10-R10-M10-L10
rm tmp/test1-r10-R10-M10-L10.bk
rm tmp/test2-r10-R10-M10-L10.bk
rm tmp/test1ona
rm tmp/test2ona
