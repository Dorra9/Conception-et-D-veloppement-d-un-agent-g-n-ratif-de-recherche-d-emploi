import numpy as np
import json
import re
import tensorflow as tf
import random
import spacy
nlp = spacy.load('en_core_web_sm')
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
import profile_scrapping 

import random
# importing pandas library
import pandas as pd
import requests, time, random
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import pyautogui as pag
import csv
import pandas as pd
import nltk
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
url =  "http://linkedin.com/"
network_url =  "https://www.linkedin.com/in/dora-moatamri-b3b7b3187/"
driver = webdriver.Chrome("C:/Users/ASUS/Downloads/chromedriver_win32 (4)/chromedriver.exe")
  
# reading given csv file 
# and creating dataframe
df = pd.read_csv("C:/Users/ASUS/Desktop/intents/intents.txt")

  
# adding column headings
df.columns = ['sentences', 'intent']
intent = df["intent"]
unique_intent = list(set(intent))
sentences = list(df["sentences"])
  
# store dataframe into csv file
df.to_csv('intents.csv', 
                index = None)
text = list(map(str.split,sentences))
from gensim.models import fasttext
models = fasttext.FastText() # Using default configuration. You can set size, window, min_count, etc. Refer https://radimrehurek.com/gensim/models/fasttext.html
models.build_vocab(text)
models.train(text, total_examples=len(text), epochs=20)
models.build_vocab(text, update=True)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast 
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.corpus import wordnet
# from surprise import Reader, Dataset, SVD, evaluate
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec

import pickle



import gc
import re

import warnings; warnings.simplefilter('ignore')

def preprocessor(text):
    text = text.replace('\\r', '').replace('&nbsp', '').replace('\n', '')
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text



def user_exist(user):
    if len(users.loc[users['Respondent'] == user]) == 0:
        return False
    return True

def has_coordinates(user):
    c = users.loc[users['Respondent'] == user, "Coordinates"] 

    if (len(c) == 0) or (c.iloc[0].split(',')[0]) == 'None':
        return False
    return True



def historical_application(user_id):
    historical_apps = users.loc[users.Respondent == user_id]['Respondent']
    content = list()
    for application in historical_apps:
        temp = jobs1.loc[jobs1.jobid == application, [ 'skills']]
        if len(temp) != 0:
            content += [temp.jobtitle.values + ". " + temp.jobdescription.values + ". " + temp.skills.values]
    return content


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))
        self.epoch += 1

jobs1 = pd.read_csv("C:/Users/ASUS/Desktop/Job-Recommendation-System-master/Job-Recommendation-System-master/data/collaborative filtering/dice_com-job_us_sample.csv")
users=pd.read_csv("C:/Users/ASUS/Desktop/Job-Recommendation-System-master/Job-Recommendation-System-master/data/collaborative filtering/survey_results_public.csv")

jobs1['jobdescription'] = jobs1['jobdescription'].astype(dtype='str').apply(preprocessor)
jobs1['skills'] = jobs1['skills'].astype(dtype='str').apply(preprocessor)
jobs1['profile'] =  jobs1['skills'].astype(str)
jobs = jobs1[['jobid','profile']]


def similarities_nlp_model( model_name = "jobs_doc2vec_model",
                 mapping_name = "jobID_mapping.p", max_epochs = 100,
                 alpha = 0.025):
    
    document = list()
    jobID_mapping = dict()
        
    for i, token in enumerate(jobs["profile"]):
        value = jobs.iloc[i]["jobid"]
        tokens = TaggedDocument(simple_preprocess(token), [i])
        document.append(tokens)
        jobID_mapping[i] = value

    epoch_logger = EpochLogger()
    model = Doc2Vec(size = 20, alpha=alpha, 
                    min_alpha=0.00025, min_count=2,
                    callbacks=[epoch_logger], dm =1, workers=8, window=2)
    
    model.build_vocab(document)
    print(model.corpus_count)
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(document, 
                    total_examples=model.corpus_count, 
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    # create a dictionary
    pickle.dump(model, open(model_name, "wb")) 
    pickle.dump(jobID_mapping, open(mapping_name, "wb")) 

# -------------------------------------------------------------
# Load the dictionary back from the pickle file.
    return model, jobID_mapping
file = open("C:/Users/ASUS/Desktop/pfe project/jobID_mapping",'rb')
jobID_mapping = pickle.load(file)
mapping_name=open("C:/Users/ASUS/Desktop/pfe project/jobs_doc2vec_model",'rb')
model=pickle.load(mapping_name)


models.train(text, total_examples=len(text), epochs=models.epochs) 
data_file = open('C:/Users/ASUS/Desktop/dora.txt').read()
intents = json.loads(data_file)
import re
regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
def classifier (user_utterance):
    
    threshold = 0.3 # This can changed as per the need.
    max_similarity = 0
    for i in range(len(sentences)):
        if models.wv.similarity(sentences[i], user_utterance) > max_similarity:
            max_similarity = models.wv.similarity(sentences[i], user_utterance)
        #print(max_similarity)
            max_intent = intent[i]
        #print(max_intent)
    #if max_similarity < threshold: # if for a user utterance, the maximum similarity is less than the threshold, we classify as it as unknown intent
           # print("Didn't get it. Can you please type it again?")
   # else:
            #print("The detected intent is:", (max_intent))
    return(max_intent)

def get_Response(c,intents_json):
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
    res = get_Response(classification, intents)
    return res

def getResponse(user_response , final_jobs):
    
    flag=True
    Response="Here to help you to find a job ! Please put your linkedin adress or your CV! "
    
    if(user_response!=''):
        
        
        if (((re.match(regex, user_response) is not None))==True):
            print("ok")
            flag=False
            driver.get(user_response)
            SCROLL_PAUSE_TIME = 5
            time.sleep(40)

            src = driver.page_source
            soup = BeautifulSoup(src, 'html')
            global recommendations

            recommendations =profile_scrapping.content_distance_based_recommender(soup, jobID_mapping , model , top = 10)
            
            for row in recommendations .itertuples():
                    
                    Response = "the most job that much your profile is:"+ row.jobtitle+ "   mach your profile with a score of :"+str((row.score))
        
        #elif(verifier_pdf(user_response==vrai)):
                #flag=False
                #dfR=get_recomendation(user_response)
                #Response="Here some links that much with your CV:  "+convert_list_to_string(dfR["link"].values,seperator=',')
            
        elif (classifier(user_response) == "recommendation"):
            
            Response= "Here to help you to find a job ! Please put your linkedin adress or your CV!"
            
            flag=False
        elif (classifier(user_response) == "score"):
                 #Response = "the most job that much your profile is:  "+ ((convert_list_to_string(str(recommendation ["jobtitle"].values),seperator=','))) + "much your profile with a score of "+( (convert_list_to_string(recommendation ["jobtitle"].values,seperator=',')))
                for row in recommendations .itertuples():
                    Response= ("the most job that much your profile is:"+ row.jobtitle+ "   mach your profile with a score of :"+str((row.score)))
                            
        elif (classifier(user_response) == "skills"):
                for row in recommendations .itertuples():
                     Response= ("the skills that this job demand are:  "+ row.skills)
                            
        elif (classifier(user_response) == "jobdescription"):
            for row in recommendations .itertuples():
                    Response= ("Here a description about this job:  "+ row.jobdescription)
                     
                
        elif (classifier(user_response) == "company"):
            for row in recommendations .itertuples():
                    Response= ("the name of the company is:  "+ row.company)
                    
                
            
                
        elif (classifier(user_response) == "process"):
            for row in recommendations .itertuples():
                    Response= ("To apply for this job you should go to ehis URL:  "+ row.advertiserurl)
                
        else :
                
                Response=chatbot_response(user_response)
                
                
    return Response


                

