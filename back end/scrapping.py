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
driver = webdriver.Chrome("C:/Users/ASUS/Downloads/chromedriver_win32 (3)/chromedriver.exe")


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
#print ('Enter the linkedin adress')
#link =input()
#driver.get(link)
SCROLL_PAUSE_TIME = 5
#time.sleep(40)



    
#src = driver.page_source
#soup = BeautifulSoup(src, 'html')

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

user_view = pd.read_csv("C:/Users/ASUS/Desktop/Job-Recommendation-System-master/Job-Recommendation-System-master/data/collaborative filtering/survey_results_public.csv")

user_view = user_view[['Respondent', 'DevType', 'Employment', 'JobSearchStatus','CommunicationTools','DatabaseWorkedWith']]

user_view["profile"] = user_view["DevType"] +"  "+ user_view["Employment"]+"  "+ user_view["JobSearchStatus"]+"  "+ user_view["CommunicationTools"]+"  "+ user_view["DatabaseWorkedWith"]

user_view['profile'] = user_view['profile'].str.replace('[^a-zA-Z \n\.]',"")

user_view['profile'] = user_view['profile'].str.lower()

user_view = user_view[['Respondent','profile']]

user_view['profile'] = user_view['profile'].str.replace('[^a-zA-Z \n\.]'," ") 

user_view['profile'] = user_view['profile'].str.lower()

final_all1 = user_view[['Respondent', 'profile']]
# renaming the column name as it seemed a bit complicated
final_all1 = user_view[['Respondent', 'profile']]
final_all1 = final_all1.fillna(" ")

pos_com_city = final_all1['profile']
#removing stopwords and applying potter stemming
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stemmer =  PorterStemmer()
stop = stopwords.words('english')
only_text = pos_com_city.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#adding the featured column back to pandas
final_all1['text']= only_text
# As we have added a new column by performing all the operations using lambda function, we are removing the unnecessary column
#final_all = final_all.drop("pos_com_city_empType_jobDesc", 1)

list(final_all1)


 
#time.sleep(10)   
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
    competences = soup.find('section', {'class': 'pv-profile-section pv-skill-categories-section artdeco-card mt4 p5 ember-view'})
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
    
    skil=getskills(soup)
    output = pd.DataFrame ({'Name': name, 'Profile': profilee , 'Location': locationn, 'Company':companyy , 'Job': Job_title,'Experience':Experience,'Date_de travail':date,'skills':skil}, index=[0])
    output['skills']=output['skills'].astype(dtype='str')
    output['Job'] = output['Job'].astype(dtype='str')
    output['profile'] =  (output['Job'].astype(str) +  " " ) + (output['skills'].astype(str) + " ")

    print(output)
    user_q=output['profile']
    
    return user_q





def get_recommendation(soup, df_all):
    usesr_q=getprofile(soup)
    user_tfidf = tfidf_vectorizer.transform(usesr_q)
    cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_jobid)

    output2 = list(cos_similarity_tfidf)
    top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
    list_scores = [output2[i][0][0] for i in top]
    recommendation = pd.DataFrame(columns = [ 'jobid',  'jobtitle', 'score'])
    count = 0
    for i in top:
        #recommendation.at[count, 'ApplicantID'] = 
        recommendation.at[count, 'jobid'] = df_all['jobid'][i]
        recommendation.at[count, 'jobtitle'] = df_all['jobtitle'][i]
        recommendation.at[count, 'score'] =   list_scores[count]
        count += 1
    result = recommendation.to_json(orient="split")
    parsed = json.loads(result)
    result=json.dumps(parsed, indent=4)
    return result


