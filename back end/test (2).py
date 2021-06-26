from flask import Flask, render_template, jsonify, request
#import scrapping
import Intent_Classification_and_Prediction
import requests, time, random
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import pyautogui as pag
from gensim.models.callbacks import CallbackAny2Vec
import csv
import pandas as pd
import scr1
from flask_cors import CORS
#import recomandation



app = Flask(__name__)
CORS(app)
class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))
        self.epoch += 1

@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    final_jobs = pd.read_csv("C:/Users/ASUS/Desktop/Job-Recommendation-System-master/Job-Recommendation-System-master/data/collaborative filtering/dice_com-job_us_sample.csv")    
    
    if request.method == 'POST':
        
        
        
        result=request.json
        
        the_question=result['the_question']
        
         

        
    response = scr1.getResponse(the_question, final_jobs)

    return jsonify({"response": response })










if __name__ == '__main__':
    app.run(debug=True)