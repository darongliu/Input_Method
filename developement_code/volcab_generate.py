#! /usr/bin/python
import codecs
all_word = set()
count = 0
with codecs.open('../data/all_chinese.ptt.corpus.20140906.txt','r',encoding = 'utf8') as f:
    for line in f:
        word = line.split()
        for one_word in word:
            all_word.add(one_word)
        print count
        count = count+1
print len(all_word)

count = 0
with codecs.open('../data/vocab','w',encoding = 'utf8') as f:
    for word in all_word:
        f.write(word)
        f.write(' ')
        f.write(str(count))
        f.write('\n')
        count = count + 1
