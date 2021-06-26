import csv
import re
import spacy
import sys
import pandas as pd
from io import StringIO
import pandas
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import os
import sys, getopt
import numpy as np
from bs4 import BeautifulSoup
from urllib.request import urlopen
tfidf_vectorizer = TfidfVectorizer()
#Function converting pdf to string
final_jobs = pd.read_csv("C:/Users/ASUS/Desktop/Job-Recommendation-System-master/Job-Recommendation-System-master/data/collaborative filtering/dice_com-job_us_sample.csv")

final_jobs["profile"] = final_jobs["joblocation_address"].map(str) + " " + final_jobs["company"] +" "+ final_jobs["jobtitle"]+ " "+final_jobs['employmenttype_jobstatus']+" "+final_jobs['jobdescription']+" "+final_jobs['skills']

#converting all the characeters to lower case
final_jobs['profile'] = final_jobs['profile'].str.lower() 

final_all = final_jobs[['jobid', 'profile']]
# renaming the column name as it seemed a bit complicated
final_all = final_jobs[['jobid', 'profile']]
final_all = final_all.fillna(" ")
pos_com_city_empType_jobDesc = final_all['profile']
#removing stopwords and applying potter stemming
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stemmer =  PorterStemmer()
stop = stopwords.words('english')
only_text = pos_com_city_empType_jobDesc.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

only_text = only_text.apply(lambda x : filter(None,x.split(" ")))

only_text = only_text.apply(lambda x : [stemmer.stem(y) for y in x])

only_text = only_text.apply(lambda x : " ".join(x))

final_all['text']= only_text
# As we have added a new column by performing all the operations using lambda function, we are removing the unnecessary column
#final_all = final_all.drop("pos_com_city_empType_jobDesc", 1)

list(final_all)

tfidf_vectorizer = TfidfVectorizer()

tfidf_jobid = tfidf_vectorizer.fit_transform((final_all['text'])) #fitting and transforming the vector
def convert(fname, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = open(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close
    return text
#Function to extract names from the string using spacy
def extract_name(string):
    r1 = str(string)
    nlp = spacy.load('xx_ent_wiki_sm')
    
    doc = nlp(r1)
    for ent in doc.ents:
        if(ent.label_ == 'PER'):
            print(ent.text)
            break
    return (ent.text)
#Function to extract Phone Numbers from string using regular expressions
def extract_phone_numbers(string):
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', number) for number in phone_numbers]
#Function to extract Email address from a string using regular expressions
def extract_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)


def extract_skills(path):
    resume_string = convert(path)
    resume_string1 = resume_string
    #Removing commas in the resume for an effecient check
    resume_string = resume_string.replace(',',' ')
    #Converting all the charachters in lower case
    resume_string = resume_string.lower()
    
    with open('techatt.csv', "r") as f:
        reader = csv.reader(f)
        your_listatt = list(reader)
    with open('techskill.csv', "r") as f:
        reader = csv.reader(f)
        your_list = list(reader)
    with open('nontechnicalskills.csv', "r") as f:
        reader = csv.reader(f)
        your_list1 = list(reader)
#Sets are used as it has a a constant time for lookup hence the overall the time for the total code will not exceed O(n)

    s = set(your_list[0])
    s1 = your_list
    s2 = your_listatt
    skillindex = []
    global Strr
    skills = []
    final=[]
    skillsatt = []
    for word in resume_string.split(" "):
        if word in s:
          
            skills.append(word)
            for i in skills :
                final.append(i)
                final=[s.replace(' ', '') for s in final]
                Str = " ".join(final)
           
                Strr = " ".join(skills)
            
    return Strr


            
def get_profile(string):       

    nom = extract_name(string)
    y = extract_phone_numbers(string)
    email=extract_email_addresses(string)
    Skils=extract_skills(string)
     
   
    output = pandas.DataFrame ({'Skills':Skils}, index=[0])
    

    output['profile'] = (output['Skills'].astype(str) )
    user=output['profile']
    #print(user_q)
    
    return user



def get_recommendation(resume_string, df_all):
    usesr_q=get_profile(resume_string)
    user_tfidf = tfidf_vectorizer.transform(usesr_q)
    cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_jobid)

    output2 = list(cos_similarity_tfidf)
    top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
    list_scores = [output2[i][0][0] for i in top]
    recommendation = pd.DataFrame(columns = [ 'jobid', 'jobtitle','jobdescription' ,'skills','company','score','advertiserurl'])
    count = 0
    for i in top:
        #recommendation.at[count, 'ApplicantID'] = 
        recommendation.at[count, 'jobid'] = df_all['jobid'][i]
        recommendation.at[count, 'jobtitle'] = df_all['jobtitle'][i]
        recommendation.at[count, 'jobdescription'] = df_all['jobdescription'][i]
        recommendation.at[count, 'skills'] = df_all['skills'][i]
        recommendation.at[count, 'advertiserurl'] = df_all['advertiserurl'][i]
        recommendation.at[count, 'company'] = df_all['company'][i]
        recommendation.at[count, 'score'] =   list_scores[count]
        count += 1
    #result = recommendation.to_json(orient="split")
    #parsed = json.loads(result)
    #result=json.dumps(parsed, indent=4)
    return recommendation

