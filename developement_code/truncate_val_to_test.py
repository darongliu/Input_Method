import os
import random
import re

val_file = "../data/all_chinese_char_seperate_val.ptt.corpus.20140906.txt"
test_file= "../data/all_chinese_char_seperate_test.ptt.corpus.20140906.txt"
test_file_ans = "../data/all_chinese_char_seperate_test_ans.ptt.corpus.20140906.txt"

with open(val_file , 'r') as f_read , open(test_file , 'w') as f_test  , open(test_file_ans , 'w') as f_ans:
    all_lines = f_read.readlines()
    for line in all_lines :
        count = 0
        words = line.split()
        if len(words) <= 5 :
            continue
        else : 
            pass
        while count < 2 : 
            num = random.randrange(4,len(words),1)
            if words[num] == "o" :
                count = count + 1
            else :
                break
      
        if count < 2 :
            for i in range(num+1) :
                f_test.write(words[i])    
                f_test.write(" ")
            f_test.write("\n")
          
            for i in range(len(words)) :
                f_ans.write(words[i])
                f_ans.write(" ")
            f_ans.write("\n")

        else :
            pass
        
         
