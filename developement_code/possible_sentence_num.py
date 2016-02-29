import os

sen_root = "../data/all_test_possible_sentence"
save = "../data/possible_sen_num"

all_num = 1386

with open(save , 'w') as f_write:
    for i in range(all_num) :
        dir_path = os.path.join(sen_root , "sentence_"+str(i))
        all_file = os.listdir(dir_path)
        f_write.write(str(len(all_file)))
        f_write.write("\n")
