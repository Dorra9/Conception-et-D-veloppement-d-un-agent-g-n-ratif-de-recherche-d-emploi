import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast 
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

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


def similarities_nlp_model( model_name = "C:/Users/ASUS/Desktop/jobs_doc2vec_model",
                 mapping_name = "C:/Users/ASUS/Desktop/jobID_mapping", max_epochs = 100,
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
        
        model.alpha -= 0.0002
        
        model.min_alpha = model.alpha

    
    fname = get_tmpfile("C:/Users/ASUS/Desktop/jobs_doc2vec_model")
    model.save(fname)
    pickle.dump(jobID_mapping, open(mapping_name, "wb")) 


    return model, jobID_mapping

model, jobID_mapping = similarities_nlp_model()


