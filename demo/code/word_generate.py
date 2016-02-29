# -*- coding: UTF-8 -*-
import subprocess
import os

#one symbol represents many chuyin #TODO
key_pair = {}
key_pair['a'] = ['ㄅ','ㄆ','ㄇ','ㄈ']
key_pair['b'] = ['ㄉ','ㄊ','ㄋ','ㄌ']
key_pair['c'] = ['ㄍ','ㄎ','ㄏ','ㄐ','ㄑ','ㄒ','ㄓ','ㄔ']
key_pair['d'] = ['ㄕ','ㄖ' ,'ㄗ','ㄘ','ㄙ','ㄧ']
key_pair['e'] = ['ㄨ','ㄩ','ㄚ']
key_pair['f'] = ['ㄛ','ㄜ','ㄝ','ㄞ']
key_pair['g'] = ['ㄟ','ㄠ','ㄡ','ㄢ']
key_pair['h'] = ['ㄣ','ㄤ','ㄥ','ㄦ']

all_key = key_pair.keys()

#read data from vocab file and map file

vocab_file = '../data/char'#TODO
map_file = '../data/Utf8-ZhuYin.map'#TODO

with open(vocab_file , 'rb') as f:
    all_lines = f.readlines()
    exist_vocab = set()
    for line in all_lines :
        parse = line.split()
        exist_vocab.add(parse[0])
with open(map_file , 'rb') as f:
    all_lines = f.readlines()
    all_vocab = dict()
    for line in all_lines :
        parse = line.split()
        all_vocab[parse[0]] = parse[1].split("/")

def possible_generate(pre_sentence , word) :

    #incorrect input
    for char in word :
        if char not in all_key :
            return False , []

    #generate all possible combination
    all_possible_combination = []
    for char in word :
        chuyin_temp = key_pair[char]
        if len(all_possible_combination) == 0 :
            all_possible_combination = chuyin_temp 
        else :
            temp = []
            for combination in all_possible_combination :
                for chuyin in chuyin_temp :
                    temp.append(combination+chuyin)
            all_possible_combination = temp

    #from map file find all possible word
    possible_word = []
    for combination in all_possible_combination :
        for vocab in exist_vocab :
            try :
                all_possible_chuyin = all_vocab[vocab]
            except :
                pass
            else :
                for chuyin in all_possible_chuyin :
                    position = chuyin.find(combination)
                    if position == 0 :
                        possible_word.append(vocab)

    if len(possible_word) == 0 :
        return True , []

    #add possible word to pre_sentence and write it to a file
    possible_sentence_file = '../tmp/possible_sentence'#TODO
    with open(possible_sentence_file , 'wb') as f_write :
        if len(pre_sentence) == 0 :
            for word in possible_word :
                sentence = 'o' + pre_sentence + word
		sentence = sentence.decode('utf8')
		sentence = ' '.join(sentence)
                f_write.write(sentence.encode('utf8'))
                f_write.write('\n')
        else :
            for word in possible_word :
                sentence = pre_sentence + word
		sentence = sentence.decode('utf8')
		sentence = ' '.join(sentence)
                f_write.write(sentence.encode('utf8'))
                f_write.write('\n')

    #fork language model to test the target file #TODO
    write_file = '../tmp/result'
    command = '../../rwthlm/rwthlm --vocab ../data/char --debug-no-sb --ppl ' + possible_sentence_file + ' --word-wrapping verbatim --verbose  ../data/ptt_language_model-i300-m300 > ' + write_file
    subprocess.call(command , shell=True)

    #parse the possibility from the output result
    all_ppl = []
    with open(write_file , 'rb') as f :
        #parse it and save it into all_ppl
        all_data = f.read().splitlines()
        if len(pre_sentence) == 0 :
            sentence_length = 1
        else :
            sentence_length = len(pre_sentence.decode('utf8'))
        start  = False
        count  = 1
        for line in all_data :
            if start is False :
                if line.strip() == 'perplexity:' : 
                    start = True

            else :
                if count == sentence_length :
                    parse = line.split()
		    try :
                        ppl = float(parse[7])
                    except :
                        pass
                    else :
                        all_ppl.append(ppl)
                    count = 1
                else :
                    count = count + 1

    assert (len(possible_word) == len(all_ppl))
        
    def ppl_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: seq[x] , reverse = True)

    sorted_idx = ppl_argsort(all_ppl)
    sorted_possible_word = [possible_word[i] for i in sorted_idx]

    return True , sorted_possible_word
