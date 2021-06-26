from flask import Flask, render_template, jsonify, request
#import scrapping
import Intent_Classification_and_Prediction
import requests, time, random
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import pyautogui as pag
import csv
import pandas as pd
from flask_cors import CORS
#import recomandation



app = Flask(__name__)
CORS(app)


@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    
    
    if request.method == 'POST':
        
        
        
        result=request.json
        
        the_question=result['the_question']
        
         

        
    response = Intent_Classification_and_Prediction.chatbot_response(the_question)

    return jsonify({"response": response })










if __name__ == '__main__':
    app.run(debug=True)