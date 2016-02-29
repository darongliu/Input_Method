import os

data_root = "../data"
origin_file = os.path.join(data_root , "all_chinese_char_seperate_train.ptt.corpus.20140906.txt")
save_file = os.path.join(data_root , "all_chinese_char_seperate_train_short.ptt.corpus.20140906.txt")

with open(origin_file , 'r') as f_read , open(save_file , 'w') as f_save :
    alllines = f_read.readlines()
    for line in alllines:
        words = line.split()
        if len(words) < 20 :
            for word in words:
                f_save.write(word)
                f_save.write(" ") 
            f_save.write("\n")

