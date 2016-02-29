def isChinese(word):
    for i in range(len(word)):
        if word[i] >= u'\u4e00' and word[i]<=u'\u9fa5':
            continue
        else:
            return False
    return True

import codecs

if __name__ == '__main__':
    count = 0
    temp = []
    with codecs.open('../data/ptt.corpus.20140906.txt','r',encoding = 'utf8') as f:
        with codecs.open('../data/all_chinese_char_seperate.ptt.corpus.20140906.txt','w',encoding = 'utf8') as f1:
            for line in f:

                if count % 1000 == 0:
                    print(count)
                count = count+1

                words = line.split()
                for one_word in words:
                    if not isChinese(one_word) :
                        temp.append("others")
                    else :
                        temp.append(one_word)
                for one_word in temp:
                    if one_word == 'others':
                        f1.write('o')
                        f1.write(' ')
                    else :
                        for char in one_word:
                            f1.write(char)
                            f1.write(' ')
                f1.write('\n')
                temp = []

