with open('../data/all_chinese_char_seperate_test.ptt.corpus.20140906.txt' , 'r') as f_read , open('../data/test_num' , 'w') as f_write:
    all_lines = f_read.readlines()
    for line in all_lines :
        word = line.split()
        f_write.write(str(len(word)))
        f_write.write("\n")
