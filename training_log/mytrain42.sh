#! /bin/bash
../rwthlm-0.11/rwthlm --momentum 0.0095 --vocab ../data/char --train ../data/all_chinese_char_seperate_train.ptt.corpus.20140906.txt --dev ../data/all_chinese_char_seperate_val.ptt.corpus.20140906.txt --learning-rate 0.001 --batch-size 100 --sequence-length 2000  --max-epoch 100000 ./tmp/pttlanguagemodel-i300-m300
