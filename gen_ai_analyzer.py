"""
Generative AI Analysis Module
Uses LLMs for detailed fraud analysis and explanations
"""

import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class GenAIAnalyzer:
    """
    Uses Claude/ChatGPT API for advanced analysis
    """
    
    def __init__(self, model_type: str = "google"):
        """
        Initialize with selected LLM
        model_type: "openai", "google", or "anthropic"
        """
        self.model_type = model_type
        self.setup_client()
    
    def setup_client(self):
        """Setup the appropriate LLM client"""
        if self.model_type == "google":
            try:
                import google.generativeai as genai
                self.client = genai.Client()
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))
                self.model = "gemini-pro"
            except ImportError:
                print("Google Generative AI not installed")
        
        elif self.model_type == "openai":
            try:
                import openai
                self.client = openai
                self.client.api_key = os.getenv("OPENAI_API_KEY", "")
                self.model = "gpt-3.5-turbo"
            except ImportError:
                print("OpenAI not installed")
    
    def generate_fraud_analysis(self, 
                               title: str, 
                               company: str, 
                               location: str,
                               salary: str,
                               description: str,
                               prediction: int,
                               confidence: float) -> Dict[str, Any]:
        """
        Generate detailed fraud analysis using LLM
        """
        
        prompt = f"""Analyze this job posting for potential fraud indicators:

Job Title: {title}
Company: {company}
Location: {location}
Salary: {salary}
Description: {description}

Initial AI Prediction: {'FAKE' if prediction == 1 else 'REAL'} (Confidence: {confidence}%)

Please provide:
1. Specific fraud red flags found
2. Legitimate aspects identified
3. Risk assessment (Low/Medium/High)
4. Recommendations for job seeker

Format your response as JSON with keys: red_flags, positive_aspects, risk_level, recommendations
"""
        
        try:
            if self.model_type == "google":
                response = self._analyze_with_google(prompt)
            elif self.model_type == "openai":
                response = self._analyze_with_openai(prompt)
            else:
                response = self._analyze_with_local(prompt)
            
            return response
        
        except Exception as e:
            return {
                "error": str(e),
                "red_flags": [],
                "positive_aspects": [],
                "risk_level": "UNKNOWN",
                "recommendations": ["Error analyzing job posting"]
            }
    
    def _analyze_with_google(self, prompt: str) -> Dict[str, Any]:
        """Analyze using Google Generative AI"""
        try:
            import google.generativeai as genai
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            
            # Parse response
            response_text = response.text
            
            # Try to extract JSON from response
            try:
                # Find JSON in response
                import json
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    return json.loads(json_str)
            except (ValueError, json.JSONDecodeError):
                pass
            
            return {
                "analysis": response_text,
                "red_flags": [],
                "positive_aspects": [],
                "risk_level": "MEDIUM",
                "recommendations": response_text.split('\n')[:3]
            }
        
        except Exception as e:
            raise Exception(f"Google API Error: {str(e)}")
    
    def _analyze_with_openai(self, prompt: str) -> Dict[str, Any]:
        """Analyze using OpenAI API"""
        try:
            import openai
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            
            response_text = response['choices'][0]['message']['content']
            
            try:
                import json
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    return json.loads(json_str)
            except (ValueError, json.JSONDecodeError):
                pass
            
            return {
                "analysis": response_text,
                "red_flags": [],
                "positive_aspects": [],
                "risk_level": "MEDIUM",
                "recommendations": response_text.split('\n')[:3]
            }
        
        except Exception as e:
            raise Exception(f"OpenAI API Error: {str(e)}")
    
    def _analyze_with_local(self, prompt: str) -> Dict[str, Any]:
        """Fallback local analysis using heuristics"""
        # This is a fallback when no API is configured
        return {
            "analysis": "Local analysis mode - configure API keys for LLM analysis",
            "red_flags": [],
            "positive_aspects": [],
            "risk_level": "MEDIUM",
            "recommendations": ["Configure LLM API to enable detailed analysis"]
        }
    
    def generate_explanation(self, 
                            title: str,
                            prediction: int,
                            confidence: float,
                            red_flags: list) -> str:
        """Generate human-readable explanation"""
        
        prompt = f"""Provide a brief, user-friendly explanation for why this job posting might be {'FAKE' if prediction == 1 else 'REAL'}:

Job Title: {title}
Confidence: {confidence}%
Red Flags Detected: {', '.join(red_flags) if red_flags else 'None'}

Keep explanation under 2 sentences, using simple language.
"""
        
        try:
            if self.model_type == "google":
                import google.generativeai as genai
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                return response.text
            elif self.model_type == "openai":
                import openai
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=150
                )
                return response['choices'][0]['message']['content']
        
        except Exception as e:
            return f"Prediction: {'Fake' if prediction == 1 else 'Real'} Job Posting ({confidence}% confidence)"
    
    def generate_report(self, job_data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate comprehensive fraud analysis report"""
        
        report = f"""
========== FRAUD ANALYSIS REPORT ==========

Job Title: {job_data.get('title', 'N/A')}
Company: {job_data.get('company', 'N/A')}
Location: {job_data.get('location', 'N/A')}

RISK ASSESSMENT: {analysis.get('risk_level', 'UNKNOWN')}

RED FLAGS IDENTIFIED:
{self._format_list(analysis.get('red_flags', []))}

POSITIVE ASPECTS:
{self._format_list(analysis.get('positive_aspects', []))}

RECOMMENDATIONS:
{self._format_list(analysis.get('recommendations', []))}

==========================================
"""
        return report
    
    def _format_list(self, items: list) -> str:
        """Format list items for report"""
        if not items:
            return "- None"
        return '\n'.join([f"- {item}" for item in items])


if __name__ == "__main__":
    analyzer = GenAIAnalyzer(model_type="google")
    
    sample_job = {
        'title': 'Work from Home - Easy Money!',
        'company': 'XYZ Corp',
        'location': 'Remote',
        'salary': '$5000 - $10000',
        'description': 'URGENT! No experience needed. Make passive income easily!'
    }
    
    analysis = analyzer.generate_fraud_analysis(
        **sample_job,
        prediction=1,
        confidence=85.5
    )
    
    print(json.dumps(analysis, indent=2))
