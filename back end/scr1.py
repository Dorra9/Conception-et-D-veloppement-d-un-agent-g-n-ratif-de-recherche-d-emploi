import numpy as np

import re

import random
import spacy
nlp = spacy.load('en_core_web_sm')
import nltk
nltk.download('punkt')
nltk.download('wordnet')

import re
import pickle
#import resumeparser
import numpy as np
import Intent_Classification_and_Prediction
import random
import requests, time, random
from flask import Flask, render_template, jsonify, request
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import pyautogui as pag
import csv
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from selenium.webdriver.chrome.options import Options
from gensim.models.callbacks import CallbackAny2Vec
from flask_cors import CORS
#import recomandation



app = Flask(__name__)
CORS(app)


options = Options()
options.headless = True# Last I checked this was necessary.


url =  "http://linkedin.com/"
network_url =  "https://www.linkedin.com/in/dora-moatamri-b3b7b3187/"
driver = webdriver.Chrome("C:/Users/ASUS/Downloads/chromedriver_win32 (4)/chromedriver.exe")


#print(user_q)



def login_to_linkedin(driver):
    
    username = driver.find_element_by_id("session_key")
    username.send_keys("doramoatamri@gmail.com")
    password = driver.find_element_by_id("session_password")
    password.send_keys("50971367d  ")
    driver.find_element_by_class_name("sign-in-form__submit-button").click()
    
def  start_bot(driver,url):
    driver.get(url)
    login_to_linkedin(driver)

start_bot(driver,url)


time.sleep(20)
print ('Enter the linkedin adress')
link =input()
driver.get(link)
SCROLL_PAUSE_TIME = 5
time.sleep(40)



    
src = driver.page_source
soup = BeautifulSoup(src, 'html.parser')

tfidf_vectorizer = TfidfVectorizer()
def getName(soup):
    
    name_div = soup.find('div', {'class': 'display-flex mt2 pv-top-card--reflow'})   
    try:
        name = name_div.find('li', {'class': 'inline t-24 t-black t-normal break-words'}).get_text().strip()
    except IndexError: # To ignore any kind of error
        name = 'NULL'
    except AttributeError:
        name = 'NULL'
    
    return name
def location(soup):
    try:
        name_div = soup.find('div', {'class': 'display-flex mt2 pv-top-card--reflow'})   
        location = name_div.find('li', {'class': 't-16 t-black t-normal inline-block'}).get_text().strip()
    except IndexError:
        location = 'NULL'
    except AttributeError:
        location = 'NULL'
        
    return location

def profile(soup):
    try:
        name_div = soup.find('div', {'class': 'display-flex mt2 pv-top-card--reflow'})  
        profile_title = name_div.find('h2', {'class': 'mt1 t-18 t-black t-normal break-words'}).get_text().strip()
    except IndexError:
        profile_title = 'NULL'
    except AttributeError:
        profile_title = 'NULL'  
       
    return profile

def connections(soup):
    
    try:
        name_div = soup.find('div', {'class': 'display-flex mt2 pv-top-card--reflow'})  
        c_div = name_div.find('ul', {'class': 'pv-top-card--list pv-top-card--list-bullet mt1'})
        connections = c_div.find('span', {'class': 't-16 t-bold link-without-visited-state'}).get_text().strip()
    except IndexError:
        connections = 'NULL'
    except AttributeError:
        connections = 'NULL'
        
    return connections
    

def company(soup):
    try:
        name_div = soup.find('div', {'class': 'display-flex mt2 pv-top-card--reflow'})  
        c_div = name_div.find('a', {'class': 'pv-top-card--experience-list-item'})
        company = c_div.find('span', {'class': 'text-align-left ml2 t-14 t-black t-bold full-width lt-line-clamp lt-line-clamp--multi-line ember-view'}).get_text().strip()
    except IndexError:
        company = 'NULL'
    except AttributeError:
        company = 'NULL'
        
    return(company)

def getjob(soup):
    try:
        Experience_div = soup.find('div', {'class': 'pv-entity__summary-info pv-entity__summary-info--background-section'})
        job_title = Experience_div.find('h3', {'class': 't-16 t-black t-bold'}).get_text().strip()
    except IndexError:
        job_title = 'NULL'
    except AttributeError:
        job_title = 'NULL'
    
    return job_title

def getexperience(soup):
    try:
        experience = soup. find('span', {'class': 'pv-entity__bullet-item-v2'}).get_text().strip()
    except IndexError:
        experience = 'NULL'
    except AttributeError:
        experience = 'NULL'
        
    return experience

def date_emplois(soup):
    try:
        Experience_div = soup.find('div', {'class': 'pv-entity__summary-info pv-entity__summary-info--background-section'})
        date = Experience_div. find('h4', {'class': 'pv-entity__date-range t-14 t-black--light t-normal'}).get_text().strip()
    except IndexError:
        date = 'NULL'
    except AttributeError:
        date = 'NULL' 
        
    return date


def getskills(soup):
    competences = soup.find('section', {'class': 'pv-profile-section pv-skill-categories-section artdeco-card mt4 p5 first-degree ember-view' })
    #print(competences)
    skills = competences.find_all(attrs= {'class': 'pv-skill-category-entity__name-text t-16 t-black t-bold'})
    
    final=[]
    skl=[s.text for s in skills] 
    skl=[s.replace('\n', '') for s in skl]

    for i in skl :
        final.append(i)
    final=[s.replace(' ', '') for s in final]
    Str = " ".join(final)
    
    return Str


def getprofile(soup):
    name=getName(soup)
    locationn=location(soup)
    Job_title=getjob(soup)
    profilee=profile(soup)
    Experience=getexperience(soup)
    date=date_emplois(soup)
    companyy=company(soup)
    connectionss=connections(soup)
    URL=link
    skil=getskills(soup)
    output = pd.DataFrame ({'Name': name, 'Profile': profilee , 'Location': locationn, 'Company':companyy , 'Job': Job_title,'Experience':Experience,'Date_de travail':date,'Url':URL,'skills':skil}, index=[0])
    output['skills']=output['skills'].astype(dtype='str')
    output['Job'] = output['Job'].astype(dtype='str')
    output['profile'] =  (output['Job'].astype(str) +  " " ) + (output['skills'].astype(str) + " ")

    print(output)
    user_q=output['profile']
    
    return user_q

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

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))
        self.epoch += 1

file = open("C:/Users/ASUS/Desktop/pfe project/jobID_mapping",'rb')
jobID_mapping = pickle.load(file)
mapping_name=open("C:/Users/ASUS/Desktop/pfe project/jobs_doc2vec_model",'rb')
model=pickle.load(mapping_name)

def content_distance_based_recommender(soup, jobID_mapping , model , top = 10):
    #As infer_vector produce stochastics result I made a for to save the best list
    usesr_q=getprofile(soup)
    user_profile = usesr_q
    #user_profile = simple_preprocess(user_profile)
    user_profile= usesr_q
    #print(user_profile)
    
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
    print(tops)
    return tops
regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
import os


def verifier_pdf(path):
    
    fileName, fileExtension = os.path.splitext(path)
    if(fileExtension)=='.pdf':
        verif='vrai'
    else:
        fileExtension='Faux'
        verif=fileExtension
        
    return verif

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

            recommendations =content_distance_based_recommender(soup, jobID_mapping , model , top = 10)
            
            for row in recommendations .itertuples():
                    
                    Response = "the most job that much your profile is:"+ row.jobtitle+ "   mach your profile with a score of :"+str((row.score))
        
        #elif(verifier_pdf(user_response=='vrai')):
                #flag=False
                #recommendations=resumeparser.get_recomendation(user_response,final_jobs)
                #for row in recommendations .itertuples():
                    #Response="the most job that much your profile is:"+ row.jobtitle+ "   mach your profile with a score of :"+str((row.score))
            
        elif (Intent_Classification_and_Prediction.classifier(user_response) == "recommendation"):
            
            Response= "Here to help you to find a job ! Please put your linkedin adress or your CV!"
            
            flag=False
        elif (Intent_Classification_and_Prediction.classifier(user_response) == "score"):
                 #Response = "the most job that much your profile is:  "+ ((convert_list_to_string(str(recommendation ["jobtitle"].values),seperator=','))) + "much your profile with a score of "+( (convert_list_to_string(recommendation ["jobtitle"].values,seperator=',')))
                for row in recommendations .itertuples():
                    Response= ("the most job that much your profile is:"+ row.jobtitle+ "   mach your profile with a score of :"+str((row.score)))
                            
        elif (Intent_Classification_and_Prediction.classifier(user_response) == "skills"):
                for row in recommendations .itertuples():
                     Response= ("the skills that this job demand are:  "+ row.skills)
                            
        elif (Intent_Classification_and_Prediction.classifier(user_response) == "jobdescription"):
            for row in recommendations .itertuples():
                    Response= ("Here a description about this job:  "+ row.jobdescription)
                     
                
        elif (Intent_Classification_and_Prediction.classifier(user_response) == "company"):
            for row in recommendations .itertuples():
                    Response= ("the name of the company is:  "+ row.company)
                    
                
            
                
        elif (Intent_Classification_and_Prediction.classifier(user_response) == "process"):
            for row in recommendations .itertuples():
                    Response= ("To apply for this job you should go to ehis URL:  "+ row.advertiserurl)
                
        else :
                
                Response=Intent_Classification_and_Prediction.chatbot_response(user_response)
                
                
    return Response

@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    final_jobs = pd.read_csv("C:/Users/ASUS/Desktop/Job-Recommendation-System-master/Job-Recommendation-System-master/data/collaborative filtering/dice_com-job_us_sample.csv")    
    
    if request.method == 'POST':
        
        
        
        result=request.json
        
        the_question=result['the_question']
        
         

        
    response = getResponse(the_question, final_jobs)

    return jsonify({"response": response })










if __name__ == '__main__':
    app.run(debug=True)

         


