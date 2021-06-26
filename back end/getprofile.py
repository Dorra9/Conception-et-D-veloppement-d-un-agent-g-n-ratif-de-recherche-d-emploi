from flask import Flask, render_template, jsonify, request
import profile_scrapping


app = Flask(__name__)

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'

@app.route('/chatbot', methods=["GET", "POST"])
def getprofile():
    url=input()
    
    

 

if __name__ == '__main__':
    