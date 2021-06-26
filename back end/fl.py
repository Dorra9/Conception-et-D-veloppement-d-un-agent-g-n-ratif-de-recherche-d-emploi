from flask import Flask, render_template, jsonify, request
import Intent_Classification_and_Prediction


app = Flask(__name__)

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'


@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    
    
    if request.method == 'POST':
        
        
        
        result=request.json
        
        the_question=result['the_question']
        
         

        
    response = Intent_Classification_and_Prediction.chatbot_response(the_question)

    return jsonify({"response": response })



if __name__ == '__main__':
    app.run(debug=True)

