from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    msg = data.get("message", "").lower().strip()
    
    # Simple checking logic jaise normally handle kiya jata hai
    if 'hi' in msg or 'hello' in msg or 'hey' in msg:
        reply = "Hello! Welcome to Metro Medical Center. How can I help you today? You can ask about timings, doctors, or appointments."
    
    elif 'timing' in msg or 'time' in msg or 'open' in msg:
        reply = "We are open Monday to Saturday, from 8:00 AM to 8:00 PM. Emergency room (ER) is open 24/7."
        
    elif 'doctor' in msg or 'specialist' in msg or 'list' in msg:
        reply = "Our specialists:\n- Cardiology: Dr. Robert Chen\n- Pediatrics: Dr. Sarah Jenkins\n- Orthopedics: Dr. Allen Walker"
        
    elif 'appointment' in msg or 'book' in msg or 'fees' in msg:
        reply = "For appointments, call us directly at +1-800-555-0199 or drop your details here."
        
    elif 'bye' in msg or 'exit' in msg:
        reply = "Thank you for chatting with us. Stay safe and healthy!"
        
    else:
        reply = "Sorry, I didn't get that. Please ask about doctors, timings, or appointments."
        
    return jsonify({"response": reply})

if __name__ == '__main__':
    app.run(debug=True)
