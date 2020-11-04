#########################################################
##                                                     ##
##                      importing packages             ##
##                                                     ##
#########################################################
from collections import Counter
from emoji import UNICODE_EMOJI
from html import unescape
import joblib
from lexical_diversity import lex_div as ld
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import re
from statistics import pstdev
import xml.etree.ElementTree as ET

#########################################################
##                                                     ##
##                      reding data                    ##
##                                                     ##
#########################################################
#meta
r = open('NLP/PAN20/data/raw/en/truth.txt', "r")
data = r.read().split("\n")
idk = [] #id
spreader = [] #yes or no

for line in data:
    l = line.split(":::")
    if len(l)>1:
        idk.append(l[0])
        spreader.append(l[1])

meta_data=pd.DataFrame()
meta_data["ID"]=idk
meta_data["spreader"]=spreader

#reading and concatenating tweets
pathlist = Path('NLP/PAN20/data/raw/en').glob('**/*.xml')
ids=[]
x_raw=[]
s=0
for path in pathlist:  #iterate on filea
    s = s+1
    if s%100 == 0:
      print(s)
    head, tail = os.path.split(path)
    t=tail.split(".")
    author=t[0]
    path_in_str = str(path)
    tree = ET.parse(path_in_str)
    root = tree.getroot()
    for child in root:
        xi=[]
        for ch in child:
            x_raw.append(unescape(ch.text)) #recode xml char-s
            ids.append(author)

text_data=pd.DataFrame()
text_data["ID"]=ids
text_data["Tweets"]=x_raw

#creatig and saving DF
en_data = pd.merge(meta_data, text_data, how='inner', on = 'ID')
en_data.to_csv('NLP/PAN20/data/en_data_tweetenkent.tsv', sep='\t', index=False)


#########################################################
##                                                     ##
##                feature xtraction                    ##
##                                                     ##
#########################################################

#initializnig feature DF with 1 row per author
en_data_tweet_consist = pd.DataFrame(list(zip([en_data["ID"][i*100] for i in range(int(len(en_data["ID"])/100))],
                                              [en_data["spreader"][i*100] for i in range(int(len(en_data["ID"])/100))])),
                                                                columns =['ID', "spreader"])

##########
#
# tweets length based stats of tweets per author (charachter & word)
#

#length
len_tw_char = [len(i) for i in en_data["Tweets"]]
len_tw_word = [len(i.split(" ")) for i in en_data["Tweets"]]

#SD
len_char_sd_auth = [pstdev(len_tw_char[i*100:i*100+99]) for i in range(int(len(len_tw_char)/100))]
len_word_sd_auth = [pstdev(len_tw_word[i*100:i*100+99]) for i in range(int(len(len_tw_word)/100))]

#min - max - range - mean
len_char_min_auth = [min(len_tw_char[i*100:i*100+99]) for i in range(int(len(len_tw_char)/100))]
len_word_min_auth = [min(len_tw_word[i*100:i*100+99]) for i in range(int(len(len_tw_word)/100))]

len_char_max_auth = [max(len_tw_char[i*100:i*100+99]) for i in range(int(len(len_tw_char)/100))]
len_word_max_auth = [max(len_tw_word[i*100:i*100+99]) for i in range(int(len(len_tw_word)/100))]

len_char_rng_auth = [max(len_tw_char[i*100:i*100+99])-min(len_tw_char[i*100:i*100+99]) for
                     i in range(int(len(len_tw_char)/100))]
len_word_rng_auth = [max(len_tw_word[i*100:i*100+99])-min(len_tw_word[i*100:i*100+99]) for
                     i in range(int(len(len_tw_word)/100))]

len_char_mean_auth = [np.mean(len_tw_char[i*100:i*100+99]) for i in range(int(len(len_tw_char)/100))]
len_word_mean_auth = [np.mean(len_tw_word[i*100:i*100+99]) for i in range(int(len(len_tw_word)/100))]

##########
#
# vocab variety (TTR)
#

tweets_szerz = [" ".join(list(en_data["Tweets"])[i*100:99+i*100]) for
                i in range(int(len(len_tw_char)/100))]


ttr_szerz = [ld.ttr(ld.flemmatize(i)) for i in tweets_szerz]


##########
#
# tags
#

#RT
rt_szerz = [np.sum([k == "RT" for k in i.split(" ")]) for i in tweets_szerz]

#URL
url_szerz = [np.sum([k == "#URL#" for k in i.split(" ")]) for i in tweets_szerz]

#hashtag
hsg_szerz = [np.sum([k == "#HASHTAG#" for k in i.split(" ")]) for i in tweets_szerz]

#user
user_szerz = [np.sum([k == "#USER#" for k in i.split(" ")]) for i in tweets_szerz]

#...
p_szerz = [np.sum([k[-1:] == "…" for k in i.split(" ")]) for i in tweets_szerz]

#emoj

emoj_szerz = []
for aut in tweets_szerz:
  emdb = 0
  for tok in aut.split(" "):
    for c in tok:
      emdb += c in UNICODE_EMOJI
  emoj_szerz.append(emdb)


##########
#
# writing features to DF and saving
#

en_data_tweet_consist["len_char_sd_auth"] = len_char_sd_auth
en_data_tweet_consist["len_word_sd_auth"] = len_word_sd_auth

en_data_tweet_consist["len_char_min_auth"] = len_char_min_auth
en_data_tweet_consist["len_word_min_auth"] = len_word_min_auth

en_data_tweet_consist["len_char_max_auth"] = len_char_max_auth
en_data_tweet_consist["len_word_max_auth"] = len_word_max_auth

en_data_tweet_consist["len_char_rng_auth"] = len_char_rng_auth

en_data_tweet_consist["len_word_rng_auth"] = len_word_rng_auth

en_data_tweet_consist["len_char_mean_auth"] = len_char_mean_auth
en_data_tweet_consist["len_word_mean_auth"] = len_word_mean_auth

en_data_tweet_consist["rt_szerz"] = rt_szerz

en_data_tweet_consist["url_szerz"] = url_szerz

en_data_tweet_consist["hsg_szerz"] = hsg_szerz

en_data_tweet_consist["user_szerz"] = user_szerz

en_data_tweet_consist["p_szerz"] = p_szerz

en_data_tweet_consist["emoj_szerz"] = emoj_szerz

en_data_tweet_consist["ttr_szerz"] = ttr_szerz

en_data_tweet_consist.to_pickle('NLP/PAN20/data/en_data_tweet_consist.pkl')