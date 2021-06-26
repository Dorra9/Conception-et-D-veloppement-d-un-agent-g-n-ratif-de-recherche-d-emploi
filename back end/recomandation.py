import pandas as pd
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import scrapping

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


#print(user_q)

#user_tfidf = tfidf_vectorizer.transform(output['profile'])
#cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_jobid)

#output2 = list(cos_similarity_tfidf)


def get_recommendation(output, df_all):
    
    user_tfidf = tfidf_vectorizer.transform(output['profile'])
    cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_jobid)

    output2 = list(cos_similarity_tfidf)
    top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
    scores = [output2[i][0][0] for i in top]
    
    
    recommendation = pd.DataFrame(columns = [ 'jobid',  'jobtitle', 'score'])
    count = 0
    for i in top:
        #recommendation.at[count, 'ApplicantID'] = 
        recommendation.at[count, 'jobid'] = df_all['jobid'][i]
        recommendation.at[count, 'jobtitle'] = df_all['jobtitle'][i]
        recommendation.at[count, 'score'] =  scores[count]
        count += 1
    return recommendation

#top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
#list_scores = [output2[i][0][0] for i in top]
#get_recommendation(top,final_jobs, list_scores)
