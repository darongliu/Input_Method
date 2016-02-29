#! /bin/bash
../rwthlm-0.11/rwthlm --verbose --momentum 0.0085 --vocab ../data/char --train ../data/all_chinese_char_seperate_train.ptt.corpus.20140906.txt --dev ../data/all_chinese_char_seperate_val.ptt.corpus.20140906.txt --learning-rate 0.001 --batch-size 100 --sequence-length 2000  --max-epoch 100000 --word-wrapping verbatim ptt_language_model-i300-m300
