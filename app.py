from flask import Flask, render_template, request, jsonify
import os
import sys

# Necessary packages ko check karna aur student style auto-install karna
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[*] Missing required ML libraries. Auto-installing dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "faiss-cpu", "numpy"])
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Predefined dataset list - Question and Answer mapping (Medical Center Domain)
medical_data = [
    {"q": "hi hello hey greetings", "a": "Hello! Welcome to Metro Medical Center. How can I assist you today?"},
    {"q": "what are the hospital timings working operating hours when open", "a": "We are open Monday to Saturday from 8:00 AM to 8:00 PM. Emergency services (ER) are open 24/7."},
    {"q": "how to book an appointment schedule checking registration", "a": "To book an appointment, call our direct medical helpline at +1-800-555-0199 or register at the reception desktop portal."},
    {"q": "list of available doctors specialists cardiologist pediatrician", "a": "Specialists list:\n- Cardiology: Dr. Robert Chen\n- Pediatrics: Dr. Sarah Jenkins\n- Orthopedics: Dr. Allen Walker"},
    {"q": "departments clinics emergency services blocks available", "a": "Our center fully hosts the following specialized departments: Cardiology, Pediatrics, Orthopedics, Radiology, and General Medicine."},
    {"q": "what is the checking or consulting fee pricing structure cost", "a": "General checkup consulting fee starts from $50. Specialist doctors fee can vary depending on treatment."},
    {"q": "bye exit goodbye good night thanks", "a": "Thank you for reaching out to Metro Medical Center. Take care and stay healthy!"}
]

# Loading Hugging Face Embedding model
print("[*] Loading paraphrase-MiniLM-L6-v2 model...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Extraction of questions list to encode
questions = [item["q"] for item in medical_data]
question_embeddings = model.encode(questions)

# Vector dimension size parsing
dimension = question_embeddings.shape[1]

# Building and indexing data onto FAISS vector engine
print("[*] Creating FAISS index...")
index = faiss.IndexFlatL2(dimension)
index.add(np.array(question_embeddings).astype('float32'))

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def query_handler():
    req_data = request.get_json()
    user_query = req_data.get("message", "").lower().strip()
    
    if not user_query:
        return jsonify({"response": "Please enter a valid message."})
    
    # User message vector embedding generate karna
    query_vector = model.encode([user_query])
    
    # Search logic on FAISS index matching nearest entity distance
    distances, indices = index.search(np.array(query_vector).astype('float32'), k=1)
    
    # Fetch matched index item mapping threshold criteria evaluation
    matched_idx = indices[0][0]
    match_distance = distances[0][0]
    
    # 1.5 Is general loose threshold for distance indexing mismatch logic
    if match_distance < 1.6:
        bot_reply = medical_data[matched_idx]["a"]
    else:
        bot_reply = "I'm not completely sure about that information. Could you please specify your query? You can ask about: timings, appointments, doctors list, or consulting fee."
        
    return jsonify({"response": bot_reply})

if __name__ == '__main__':
    print("[+] Medical QA FAISS Assistant active on local address.")
    app.run(debug=True)
