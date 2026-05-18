"""
Enhanced Flask App for Fake Job Detection with NLP, Gen AI, and Computer Vision
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from nlp_features import NLPFeatureExtractor
from gen_ai_analyzer import GenAIAnalyzer
from vision_analyzer import VisionAnalyzer
from preprocess import AdvancedPreprocessor

app = Flask(__name__)

# Load trained model, vectorizer, and scaler
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Try to load additional models
try:
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    feature_names = pickle.load(open("models/feature_names.pkl", "rb"))
except:
    scaler = None
    feature_names = None

# Initialize feature extractors and analyzers
nlp_extractor = NLPFeatureExtractor()
preprocessor = AdvancedPreprocessor(strategy='moderate')
vision_analyzer = VisionAnalyzer()

# Initialize Gen AI Analyzer (optional, requires API key)
try:
    gen_ai_analyzer = GenAIAnalyzer(model_type="google")
except:
    gen_ai_analyzer = None

# Home Route
@app.route("/")
def home():
    return render_template("index.html")

# Enhanced Prediction Route with NLP + Vision
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Get form values
        title = data.get("title", "").strip()
        company = data.get("company", "").strip()
        location = data.get("location", "").strip()
        salary = data.get("salary", "").strip()
        description = data.get("description", "").strip()
        website = data.get("website", "").strip()
        image_url = data.get("image_url", "").strip()

        # Preprocess text
        title_clean = preprocessor.clean_text(title)
        description_clean = preprocessor.clean_text(description)
        company_clean = preprocessor.clean_text(company)
        location_clean = preprocessor.clean_text(location)

        # Combine all text
        combined_text = f"{title_clean} {company_clean} {location_clean} {description_clean}"

        # ========== NLP FEATURE EXTRACTION ==========
        nlp_features = nlp_extractor.extract_all_features(
            title=title,
            company=company,
            location=location,
            salary=salary,
            description=description
        )

        # ========== TEXT-BASED PREDICTION ==========
        tfidf_vector = vectorizer.transform([combined_text]).toarray()

        # Prepare features for model
        if scaler is not None and feature_names is not None:
            # Include NLP features
            nlp_feature_values = np.array([
                nlp_features.get(f, 0) for f in feature_names
            ]).reshape(1, -1)
            nlp_feature_values_scaled = scaler.transform(nlp_feature_values)
            
            X_combined = np.hstack([tfidf_vector, nlp_feature_values_scaled])
            prediction = model.predict(X_combined)[0]
            
            try:
                probability = model.predict_proba(X_combined)[0]
            except:
                probability = np.array([1 - 0.7, 0.7]) if prediction == 1 else np.array([0.7, 0.3])
        else:
            # Fall back to TF-IDF only
            prediction = model.predict(tfidf_vector)[0]
            try:
                probability = model.predict_proba(tfidf_vector)[0]
            except:
                probability = np.array([0.3, 0.7]) if prediction == 1 else np.array([0.7, 0.3])

        confidence = round(max(probability) * 100, 2)

        # ========== VISION ANALYSIS ==========
        vision_results = {}
        if website:
            vision_results['website'] = vision_analyzer.validate_company_website(website)
        
        if image_url:
            vision_results['logo'] = vision_analyzer.detect_logo_spoofing(image_url, company)
            vision_results['qr_code'] = vision_analyzer.detect_qr_codes(image_url)
        
        company_verification = vision_analyzer.verify_company_information(company, location)
        vision_results['company_info'] = company_verification

        # ========== GEN AI ANALYSIS ==========
        ai_analysis = None
        if gen_ai_analyzer:
            try:
                ai_analysis = gen_ai_analyzer.generate_fraud_analysis(
                    title=title,
                    company=company,
                    location=location,
                    salary=salary,
                    description=description,
                    prediction=int(prediction),
                    confidence=confidence
                )
            except Exception as e:
                print(f"Gen AI Analysis Error: {str(e)}")

        # ========== AGGREGATE RISK SCORE ==========
        risk_score = confidence
        
        # Adjust based on vision analysis
        if vision_results.get('company_info', {}).get('is_suspicious'):
            risk_score = min(100, risk_score + 10)
        
        if vision_results.get('website', {}).get('verdict') == 'SUSPICIOUS':
            risk_score = min(100, risk_score + 5)

        # ========== FINAL RESULT ==========
        if prediction == 1 or risk_score > 60:
            result = "❌ Fake Job Posting"
            color = "red"
        else:
            result = "✅ Real Job Posting"
            color = "green"

        # Build response
        response = {
            "result": result,
            "confidence": confidence,
            "risk_score": round(risk_score, 2),
            "color": color,
            "nlp_features": {k: v for k, v in nlp_features.items() if 'suspicious' in k or 'salary' in k},
            "vision_analysis": vision_results,
            "ai_analysis": ai_analysis if ai_analysis else {}
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "result": f"Error occurred: {str(e)}",
            "confidence": 0,
            "color": "red",
            "risk_score": 0
        }), 500

# API Route for Advanced Analysis
@app.route("/analyze-advanced", methods=["POST"])
def analyze_advanced():
    """Advanced analysis with detailed breakdown"""
    try:
        data = request.get_json()
        
        # Get predictions
        title = data.get("title", "")
        company = data.get("company", "")
        description = data.get("description", "")
        
        # Extract detailed NLP features
        detailed_features = nlp_extractor.extract_all_features(
            title=title,
            company=company,
            description=description
        )
        
        # Generate AI explanation
        explanation = ""
        if gen_ai_analyzer:
            try:
                explanation = gen_ai_analyzer.generate_explanation(
                    title=title,
                    prediction=1,
                    confidence=75.0,
                    red_flags=[]
                )
            except:
                pass
        
        return jsonify({
            "detailed_features": detailed_features,
            "ai_explanation": explanation
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run App
if __name__ == "__main__":
    print("🚀 Starting Enhanced Job Detection App...")
    print("✨ Features enabled:")
    print("   - NLP Analysis: Sentiment, Linguistic, Suspicious Patterns")
    print("   - Vision Analysis: Website Validation, Logo Detection")
    print("   - Gen AI: Advanced Fraud Analysis (if API key configured)")
    app.run(debug=True, port=5000)