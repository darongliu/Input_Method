# coding=UTF-8

import cmd
import math

from word_generate import possible_generate 

def print_possible_word(all_possible_word):
    num = len(all_possible_word)
    for count in range(num) :
        if (count % 10) == 0 :
            print
            print "%3d" % count ,
        print all_possible_word[count],
        count = count + 1
    print

"""
def possible_generate(pre_sentence , word):
    correct = True
    all_possible_word = ['劉','達','融','好','帥','帥','到','讓','別','人','無','法','競','爭']
    return correct, all_possible_word
"""

class input_shell(cmd.Cmd):
    def __init__(self):
        cmd.Cmd.__init__(self)
        self.prompt = '>>> '
        self.reset()

    def reset(self):
        self.sentence = "" #input sentence up to now
        self.possible_word = [] #possible word
        self.possible_word_num = 0 #possible number
        self.state = 0 #state 0: wait for inputting chuyin
                       #state 1: wait for inputting number
    def do_quit(self, args):
        """Quits the program."""
        print "Quitting."
        raise SystemExit

    def do_reset(self, args):
        """reset the program."""
        print "resetting."
        self.reset()
        return
    
    def default(self, line):
        """parse chuyin"""
        all_word = line.split()
        if len(all_word) != 1 :
            print "Error inputting, please input again"
            return
        else :
            word = all_word[0]
            
            if self.state == 0 :
                correct, all_possible_word = possible_generate(self.sentence , word)
                num = len(all_possible_word)
                if correct == False :
                    print "Error inputting, please input again"
                    return
                elif num == 0:
                    print "No possible word, please input again"
                    return
                else :
                    self.state = 1
                    self.possible_word = all_possible_word
                    self.possible_word_num = num
                    print_possible_word(all_possible_word)
                    print 'sentence: (%s)' % self.sentence
                    print "Please choose word"
                    return

            else :
                try :
                    choose_num = int(word)
                except :
                    print "Error inputting, please input again"
                    return 
                else :
                    if choose_num > self.possible_word_num :
                        print "Too large number, only (%d) options, please input again" % self.possible_word_num
                    elif choose_num < 0 :
                        print "Negative number detect, please input again" 
                    elif choose_num == 0: 
                        print "Input chuyin again"
                        self.state = 0
                        return
                    else :
                        self.sentence = self.sentence + self.possible_word[choose_num-1]
                        print "Choose (%s)" % self.possible_word[choose_num-1]
                        print 'sentence: (%s)' % self.sentence
                        print "Input next word"
                        self.state = 0 
                        return

if __name__ == '__main__':
    input_shell().cmdloop()