from flask import Flask, request, jsonify, render_template
from chat_bot import predict_disease, symptoms

app = Flask(__name__)

def your_function(input_str: str) -> dict:
    # Dummy implementation, replace with your actual function
    return {
        "key1": f"value1_{input_str}",
        "key2": f"value2_{input_str}",
        "key3": f"value3_{input_str}",
        "key4": f"value4_{input_str}",
    }

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/chat')
def home():
    return render_template('index.html')

@app.route('/process_string', methods=['POST'])
def process_string():
    try:
        input_data = request.json
        disease = input_data["disease"]
        days =  int(input_data["days"])
        other_symptoms =  input_data["other_symptoms"]
        
        result = predict_disease(disease,other_symptoms, days )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    try:
        result = symptoms()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
