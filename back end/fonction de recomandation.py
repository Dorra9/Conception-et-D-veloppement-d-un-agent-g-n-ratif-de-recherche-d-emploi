import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast 
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec

import pickle
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
import gc
import re
from profile_scrapping import getprofile
def preprocessor(text):
    text = text.replace('\\r', '').replace('&nbsp', '').replace('\n', '')
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text

jobs1 = pd.read_csv("C:/Users/ASUS/Desktop/Job-Recommendation-System-master/Job-Recommendation-System-master/data/collaborative filtering/dice_com-job_us_sample.csv")
users=pd.read_csv("C:/Users/ASUS/Desktop/Job-Recommendation-System-master/Job-Recommendation-System-master/data/collaborative filtering/survey_results_public.csv")

jobs1['jobdescription'] = jobs1['jobdescription'].astype(dtype='str').apply(preprocessor)
jobs1['skills'] = jobs1['skills'].astype(dtype='str').apply(preprocessor)
jobs1['profile'] =  jobs1['skills'].astype(str)
jobs = jobs1[['jobid','profile']]
tfidf_vectorizer = TfidfVectorizer()

file = open("C:/Users/ASUS/Desktop/pfe project/jobID_mapping",'rb')
jobID_mapping = pickle.load(file)
mapping_name=open("C:/Users/ASUS/Desktop/pfe project/jobs_doc2vec_model",'rb')
model = Doc2Vec.load(mapping_name)
def content_distance_based_recommender(soup, jobID_mapping = jobID_mapping, model =  model, top = 10):
    #As infer_vector produce stochastics result I made a for to save the best list
    usesr_q=getprofile(soup)
    user_profile = usesr_q
    #user_profile = simple_preprocess(user_profile)
    user_profile= usesr_q
    print(user_profile)
    
    best = 0
    tops =pd.DataFrame(index = range(top), columns = ['jobid', 'jobtitle','jobdescription' ,'skills','company','advertiserurl','score'])
    #c1 = user_profile
    #print(c1)
    #c1=tfidf_vectorizer.fit_transform(c1)
    #job_distance_list = list()
    
    for i in range (1):
        inferred_vector = model.infer_vector(user_profile)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        sum_results = 0
        count = 0
        total_recom = 0
        
        
        while True:
            job_id = jobID_mapping[sims[count][0]]
            c2 = jobs.loc[jobs['jobid'] == jobID_mapping[sims[count][0]], 'profile']
            c2=tfidf_vectorizer.fit_transform(c2)
            count +=1
            #if len(c2) == 0:
                #print("Empty", sims[count][0])
                #continue
            
            #if c2.iloc[0].split(',')[0] == 'None':
                #print('None', sims[count][0])
                #continue
                
            
            
            
            

            sum_results+=sims[count][1]
            total_recom +=1
           
            if total_recom == top:
                break
        
        ##Best simulation
        if sum_results > best:
            best = sum_results
            best_sim = sims
           
            
    for i in range(top):
        recomendation = jobID_mapping[best_sim[i][0]]
        
        tops.iloc[i]['jobid', 'jobtitle','jobdescription' ,'skills','company','advertiserurl'] = np.array(jobs1.loc[jobs1['jobid'] == recomendation][['jobid', 'jobtitle','jobdescription' ,'skills','company','advertiserurl' ]])[0]
        tops.iloc[i]['score'] = round(best_sim[i][1]*100)
    
    return tops
