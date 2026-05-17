import os
import sys

# Agar textblob library nahi hai toh use auto-install karna
try:
    from textblob import TextBlob
except ImportError:
    print("[*] TextBlob library missing. Auto-installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "textblob"])
    from textblob import TextBlob

def analyze_sentiment(text):
    """
    User ke text ka sentiment analyze karne ka function.
    Polarity score: > 0 (Positive), < 0 (Negative), == 0 (Neutral)
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if polarity > 0:
        sentiment = "Positive 😊"
    elif polarity < 0:
        sentiment = "Negative 😡"
    else:
        sentiment = "Neutral 😐"
        
    return sentiment, polarity, subjectivity

def main():
    print("==================================================")
    print("    NLP Lab Task: Sentiment Analysis System       ")
    print("==================================================")
    print("Tip: Type 'exit' to close the program.\n")
    
    while True:
        user_input = input("Enter text/review to analyze: ").strip()
        
        # Exit condition
        if user_input.lower() == 'exit':
            print("\nExiting program. Goodbye!")
            break
            
        # Empty input check
        if not user_input:
            print("write a text!\n")
            continue
            
        # Sentiment analyze karna
        sentiment, score, obj = analyze_sentiment(user_input)
        
        # Result output display
        print("\n-Result- ")
        print(f" Input Text : '{user_input}'")
        print(f" Sentiment  : {sentiment}")
        print(f" Polarity   : {score}  (-1.0 to 1.0)")
        print(f" Objectivity: {obj}  (0.0 to 1.0)")
        print("\n")

if __name__ == "__main__":
    main()
