#!/bin/bash

# For feedforward models that are trained in feedforward style (i.e., the
# models are evaluated in real-time BPTT-like style), the gradient test tells
# us that the training for the very first n-gram of a sequence is correct. With
# this test, we want to verify the correctness of the training for the
# subsequent n-grams of the sequence.
#
# To this end, we train bigram LMs in feedforward style, where we once train on
# the full sequence, and once we split the sequence into two subsequences that
# are trained one after the other.
#
# The perplexities at the end and the gradient test output should be the same!

../rwthlm --debug-no-sb --feedforward --vocab v --train a1 --dev a2 --learning-rate 0.1 --batch-size 2 --max-epoch 1 --word-wrapping verbatim --no-shuffling tmp/testa-l10-l10
../rwthlm --debug-no-sb --feedforward --vocab v --train a2 --dev a2 --learning-rate 0.1 --batch-size 2 --max-epoch 2 --word-wrapping verbatim --no-shuffling tmp/testa-l10-l10

../rwthlm --debug-no-sb --feedforward --vocab v --train b  --dev a2 --learning-rate 0.1 --batch-size 2 --max-epoch 1 --word-wrapping verbatim --no-shuffling tmp/testb-l10-l10

../rwthlm --feedforward --vocab v --train a1 --batch-size 2 --word-wrapping verbatim --self-test tmp/testa-l10-l10 > tmp/testaona1
../rwthlm --feedforward --vocab v --train a1 --batch-size 2 --word-wrapping verbatim --self-test tmp/testb-l10-l10 > tmp/testbona1

diff tmp/test[ab]ona1
rm tmp/test[ab]-l10-l10{,.bk}
rm tmp/test[ab]ona1
