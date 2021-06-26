from flask import Flask, render_template, jsonify, request
import scrapping
import Intent_Classification_and_Prediction
import requests, time, random
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import pyautogui as pag
import csv
import pandas as pd
#import recomandation



app = Flask(__name__)

#final_jobs = pd.read_csv("C:/Users/ASUS/Desktop/Job-Recommendation-System-master/Job-Recommendation-System-master/data/collaborative filtering/dice_com-job_us_sample.csv")

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'

url =  "http://linkedin.com/"
network_url =  "https://www.linkedin.com/in/dora-moatamri-b3b7b3187/"
driver = webdriver.Chrome("C:/Users/ASUS/Downloads/chromedriver_win32 (3)/chromedriver.exe")

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

@app.route('/name', methods=["GET", "POST"])
def getnom():
    
    
    
    if request.method == 'POST':
        
        
        
        result=request.json
        
        link =result['link']
        
        driver.get(link)
        SCROLL_PAUSE_TIME = 5
        time.sleep(40)



    
        src = driver.page_source
        soup = BeautifulSoup(src, 'html')

   
        time.sleep(10)   
         

        
        nom = scrapping.getName(soup)

    return jsonify({"response": nom })

@app.route('/location', methods=["GET", "POST"])
def getlocation():
    
    
    
    if request.method == 'POST':
        
        
        
        result=request.json
        
        link =result['link']
        
        driver.get(link)
        SCROLL_PAUSE_TIME = 5
        time.sleep(40)



    
        src = driver.page_source
        soup = BeautifulSoup(src, 'html')

   
        time.sleep(10)   
         

        
        location = scrapping.location(soup)

    return jsonify({"response": location })

@app.route('/profile', methods=["GET", "POST"])
def getprofile():
    
    
    
    if request.method == 'POST':
        
        
        
        result=request.json
        
        link =result['link']
        
        driver.get(link)
        SCROLL_PAUSE_TIME = 5
        time.sleep(40)



    
        src = driver.page_source
        soup = BeautifulSoup(src, 'html')

   
        time.sleep(10)   
         

        
        profilee = scrapping.profile(soup)

    return jsonify({"response": profilee })


@app.route('/connections', methods=["GET", "POST"])
def getconnections():
    
    
    
    if request.method == 'POST':
        
        
        
        result=request.json
        
        link =result['link']
        
        driver.get(link)
        SCROLL_PAUSE_TIME = 5
        time.sleep(40)



    
        src = driver.page_source
        soup = BeautifulSoup(src, 'html')

   
        time.sleep(10)   
         

        
        connectionss = scrapping.connections(soup)

    return jsonify({"response": connectionss })

@app.route('/skills', methods=["GET", "POST"])
def getskl():
    
    
    
    if request.method == 'POST':
        
        
        
        result=request.json
        
        link =result['link']
        
        driver.get(link)
        SCROLL_PAUSE_TIME = 5
        time.sleep(40)



    
        src = driver.page_source
        soup = BeautifulSoup(src, 'html')

   
        time.sleep(10)   
         

        
        skills = scrapping.getskills(soup)

    return jsonify({"response": skills })

@app.route('/company', methods=["GET", "POST"])
def getcompany():
    
    
    
    if request.method == 'POST':
        
        
        
        result=request.json
        
        link =result['link']
        
        driver.get(link)
        SCROLL_PAUSE_TIME = 5
        time.sleep(40)



    
        src = driver.page_source
        soup = BeautifulSoup(src, 'html')

   
        time.sleep(10)   
         

        
        entreprise = scrapping.company(soup)

    return jsonify({"response": entreprise })

@app.route('/job', methods=["GET", "POST"])
def getjobb():
    
    
    
    if request.method == 'POST':
        
        
        
        result=request.json
        
        link =result['link']
        
        driver.get(link)
        SCROLL_PAUSE_TIME = 5
        time.sleep(40)



    
        src = driver.page_source
        soup = BeautifulSoup(src, 'html')

   
        time.sleep(10)   
         

        
        job = scrapping.getjob(soup)

    return jsonify({"response": job })

@app.route('/experience', methods=["GET", "POST"])
def getexperiencee():
    
    
    
    if request.method == 'POST':
        
        
        
        result=request.json
        
        link =result['link']
        
        driver.get(link)
        SCROLL_PAUSE_TIME = 5
        time.sleep(40)



    
        src = driver.page_source
        soup = BeautifulSoup(src, 'html')

   
        time.sleep(10)   
         

        
        experience = scrapping.getexperience(soup)

    return jsonify({"response": experience })


@app.route('/demplois', methods=["GET", "POST"])
def getdemplois():
    
    
    
    if request.method == 'POST':
        
        
        
        result=request.json
        
        link =result['link']
        
        driver.get(link)
        SCROLL_PAUSE_TIME = 5
        time.sleep(40)



    
        src = driver.page_source
        soup = BeautifulSoup(src, 'html')

   
        time.sleep(10)   
         

        
        date = scrapping.date_emplois(soup)

    return jsonify({"response": date })

@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    
    
    if request.method == 'POST':
        
        
        
        result=request.json
        
        the_question=result['the_question']
        
         

        
    response = Intent_Classification_and_Prediction.chatbot_response(the_question)

    return jsonify({"response": response })










if __name__ == '__main__':
    app.run(debug=True)