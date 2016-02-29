import os
import random

data_root = "../data"
origin_file = os.path.join(data_root , "all_chinese_char_seperate.ptt.corpus.20140906.txt")

def shuffle(array):
    copy = list(array)
    shuffle_in_place(copy)
    return copy
def shuffle_in_place(array):
    array_len = len(array)
    assert array_len > 2, 'Array is too short to shuffle!'
    for index in range(array_len):
        swap = random.randrange(array_len - 1)
        swap += swap >= index
        array[index], array[swap] = array[swap], array[index]

with open(origin_file , 'r') as f_read , open(os.path.join(data_root , "all_chinese_char_seperate_train.ptt.corpus.20140906.txt") , 'w') as f_train , open(os.path.join(data_root , "all_chinese_char_seperate_val.ptt.corpus.20140906.txt") , 'w') as f_val :
    alllines = f_read.readlines()
    randline = shuffle(alllines)
    val = randline[:1500]
    train = randline[1500:]
    for line in val :
        f_val.write(line)

    for line in train :
        f_train.write(line)

