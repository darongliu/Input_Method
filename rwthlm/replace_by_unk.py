#!/usr/bin/python

# Replace OOVs with <unk> in training data, write result to stdout.
#
# Usage: $0 <vocab> <train>

import gzip, sys

assert len(sys.argv) == 3

vocab = set()

# read vocabulary file
for line in file(sys.argv[1]):
    line = line.strip()
    vocab.add(line)

# write data to stdout
if sys.argv[2].endswith(".gz"):
    f = gzip.open(sys.argv[2])
else:
    f = open(sys.argv[2])

for line in f:
    line = line.strip()
    tokens = line.split()
    numTokens = len(tokens)

    for i in range(numTokens):
        if tokens[i] in vocab:
            sys.stdout.write(tokens[i])
        else:
            sys.stdout.write('<unk>')
        if i == numTokens - 1:
            sys.stdout.write('\n')
        else:
            sys.stdout.write(' ')
f.close()
