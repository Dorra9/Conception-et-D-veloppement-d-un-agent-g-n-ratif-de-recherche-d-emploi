import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np

import random
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint

data_file = open('C:/Users/ASUS/Desktop/dora.txt').read()
intents = json.loads(data_file)

# reading given csv file 
# and creating dataframe
df = pd.read_csv("C:/Users/ASUS/Desktop/intents/intents.txt")

  
# adding column headings
df.columns = ['sentences', 'intent']
intent = df["intent"]
unique_intent = list(set(intent))
sentences = list(df["sentences"])
  
# store dataframe into csv file
df.to_csv('intents.csv', index = None)
text = list(map(str.split,sentences))
from gensim.models import fasttext
model = fasttext.FastText() # Using default configuration. You can set size, window, min_count, etc. Refer https://radimrehurek.com/gensim/models/fasttext.html
model.build_vocab(text)
model.train(text, total_examples=len(text), epochs=20)
model.build_vocab(text, update=True)

model.train(text, total_examples=len(text), epochs=model.epochs)

def classifier (user_utterance):
    
    threshold = 0.3 # This can changed as per the need.
    max_similarity = 0
    for i in range(len(sentences)):
        if model.wv.similarity(sentences[i], user_utterance) > max_similarity:
            max_similarity = model.wv.similarity(sentences[i], user_utterance)
        #print(max_similarity)
            max_intent = intent[i]
        #print(max_intent)
    #if max_similarity < threshold: # if for a user utterance, the maximum similarity is less than the threshold, we classify as it as unknown intent
           # print("Didn't get it. Can you please type it again?")
   # else:
            #print("The detected intent is:", (max_intent))
    return(max_intent)

def getResponse(c,intents_json):
    global result
    
    print(c)
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        
        if(i['tag']== c):
            result=""
            #print('vrai')
            result = random.choice(i['responses'])
            #break
  
    return result

def chatbot_response(msg):
    classification=classifier(msg)
    res = getResponse(classification, intents)
    return res




            
