#give all possible sentence given a sentence and its Chuyin
import os
import argparse
import sys

#arg parser
parser = argparse.ArgumentParser(description='generate all possible sentence')
parser.add_argument("-v" , "--vocab", help="input vocab file path") 
parser.add_argument("-map" , "--map", help="input map file path")
parser.add_argument("-s" , "--sentence", help="input sentence path")
parser.add_argument("-save" , "--save" , help="input save dir")
parser.add_argument("-char" , "--char" , help="input user's input Chuyin")

args = parser.parse_args()
vocab_file    = vars(args)["vocab"]
map_file      = vars(args)["map"]
sentence_file = vars(args)["sentence"]
save_dir      = vars(args)["save"]
char          = vars(args)["char"]

if os.path.isdir(save_dir) :
    pass
else :
    print("save dir doesn't exist")
    sys.exit(1)

with open(vocab_file , 'r') as f:
    all_lines = f.readlines()
exist_vocab = set()
for line in all_lines :
    parse = line.split()
    exist_vocab.add(parse[0])

with open(map_file , 'r') as f:
    all_lines = f.readlines()
all_vocab = dict()
for line in all_lines :
    parse = line.split()
    all_vocab[parse[0]] = parse[1].split("/")

count = 0
with open(sentence_file , 'r') as f:
    all_lines = f.readlines()
    try :
        assert len(all_lines) == 1
    except :
        print("there can be only one line in sentence file ><|||")
        sys.exit(1)
    else :
        pass

    for word in exist_vocab:
        try :
            pro = all_vocab[word]
        except :
            continue
        else :
            pass

        for p in pro :
            if len(p) >= len(char) and char == p[:len(char)] :
                with open(os.path.join(save_dir , str(count)) , 'w') as f_save:
                    all_word = all_lines[0].split()
                    all_word.append(word)
                    for con_word in all_word :
                        f_save.write(con_word)
                        f_save.write(" ")                   
                count = count+1
                break

    
    
    

    
