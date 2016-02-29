#use generate_all_possible_sentence.py
#give a test file with multiple sentence with their last words being groundtruth
import os 
import argparse
import sys
import subprocess

def removeDir(dirPath):
    if not os.path.isdir(dirPath):
        return
    files = os.listdir(dirPath)
    try:
        for file in files:
            filePath = os.path.join(dirPath, file)
            if os.path.isfile(filePath):
                os.remove(filePath)
            elif os.path.isdir(filePath):
                removeDir(filePath)
        os.rmdir(dirPath)
    except :
        pass

#arg parser
parser = argparse.ArgumentParser(description='give a test file with multiple sentence with their last words being groundtruth')
parser.add_argument("-map" , "--map", help="input map file path")
parser.add_argument("-t" , "--test", help="input test file")
parser.add_argument("-save" , "--save" , help="input save dir")
parser.add_argument("-v" , "--vocab", help="input vocab file path")

args = parser.parse_args()
vocab_file    = vars(args)["vocab"]
map_file   = vars(args)["map"]
test_file  = vars(args)["test"]
save_dir   = vars(args)["save"]

if os.path.isdir(save_dir) :
    pass
else :
    print("save dir doesn't exist")
    sys.exit(1)

count_sen = 0

with open(test_file , 'r') as f_test  , open(map_file , 'r') as f_map:
    all_lines = f_map.readlines()
    all_word_dict = dict()
    for line in all_lines :
        parse = line.split()
        all_poun = parse[1].split("/")
        all_word_dict[parse[0]] = list(parse[1])

    all_lines = f_test.readlines()
    for line in all_lines :
        removeDir(os.path.join(save_dir , "sentence_" + str(count_sen)))
        os.mkdir(os.path.join( save_dir , "sentence_" + str(count_sen)))
  
        words = line.split()
        last_word = words[-1]
        try :
            cho = all_word_dict[last_word][0][0]
        except :
            print("no word: " , last_word , " in the dict")
            continue
        with open("./tmp" , 'w') as f_tmp:
            for i in range(len(words)-1) :
                f_tmp.write(words[i])
                f_tmp.write(" ")

        parameter = "python3 generate_all_possible_sentence.py -v " + os.path.abspath(vocab_file) +" -map " + os.path.abspath(map_file) +" -s " + os.path.abspath("./tmp") + " -save " + os.path.join(save_dir , "sentence_" + str(count_sen)) + " -char " + cho

        subprocess.call(parameter , shell=True)  
        count_sen += 1
os.remove("./tmp")
        
