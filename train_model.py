"""
Enhanced Model Training with Multiple Feature Types
Integrates NLP, linguistic, and domain features
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from nlp_features import NLPFeatureExtractor
from preprocess import AdvancedPreprocessor

# Load and Prepare Data
print("📂 Loading dataset...")
df = pd.read_csv("dataset/fake_job_postings.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Ensure required columns exist
required_cols = ['title', 'description', 'fraudulent']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    print(f"⚠️  Missing columns: {missing}")
    if 'fraudulent' not in df.columns:
        # Try to find label column
        potential_labels = [col for col in df.columns if 'fraud' in col.lower()]
        if potential_labels:
            df['fraudulent'] = df[potential_labels[0]]

# Select and prepare columns
df = df[['title', 'description', 'fraudulent']].copy()

# Fill missing values
df['title'] = df['title'].fillna('')
df['description'] = df['description'].fillna('')

print("\n🔧 Preprocessing text...")
preprocessor = AdvancedPreprocessor(strategy='moderate')
df['title'] = df['title'].apply(preprocessor.clean_text)
df['description'] = df['description'].apply(preprocessor.clean_text)

# Combine text
df['combined_text'] = df['title'] + " " + df['description']

# Extract NLP Features
print("\n🧠 Extracting NLP features...")
nlp_extractor = NLPFeatureExtractor()

nlp_features_list = []
for idx, row in df.iterrows():
    if idx % 100 == 0:
        print(f"  Processing row {idx}/{len(df)}")
    
    features = nlp_extractor.extract_all_features(
        title=row['title'],
        description=row['description']
    )
    nlp_features_list.append(features)

# Convert to DataFrame
nlp_features_df = pd.DataFrame(nlp_features_list)
print(f"✅ Extracted {len(nlp_features_df.columns)} NLP features")

# Combine with original data
df = pd.concat([df.reset_index(drop=True), nlp_features_df], axis=1)

print("\n📊 Feature Engineering...")

# TF-IDF Vectorization
print("  Creating TF-IDF vectors...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=1000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)
tfidf_vectors = vectorizer.fit_transform(df['combined_text'])

print(f"  TF-IDF shape: {tfidf_vectors.shape}")

# Prepare feature matrix
X_text = tfidf_vectors
X_nlp = nlp_features_df.fillna(0)

# Standardize NLP features
scaler = StandardScaler()
X_nlp_scaled = scaler.fit_transform(X_nlp)

# Combine features
X_combined = np.hstack([X_text.toarray(), X_nlp_scaled])

print(f"  Combined feature matrix shape: {X_combined.shape}")

y = df['fraudulent'].values

# Train-Test Split
print("\n📈 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")

# Train Models
print("\n🤖 Training models...\n")

# Model 1: Logistic Regression
print("1️⃣  Training Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"   Accuracy: {lr_accuracy*100:.2f}%")

# Model 2: Random Forest (uses only NLP features for interpretability)
print("\n2️⃣  Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"   Accuracy: {rf_accuracy*100:.2f}%")

# Choose best model
print("\n" + "="*50)
if rf_accuracy > lr_accuracy:
    best_model = rf_model
    best_name = "Random Forest"
    best_accuracy = rf_accuracy
else:
    best_model = lr_model
    best_name = "Logistic Regression"
    best_accuracy = lr_accuracy

print(f"✨ Best Model: {best_name}")
print(f"   Test Accuracy: {best_accuracy*100:.2f}%")
print("="*50)

# Detailed Classification Report
print(f"\n📋 Classification Report ({best_name}):")
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best, target_names=['Real', 'Fake']))

# Confusion Matrix
print("\n🎯 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(f"   True Negatives: {cm[0,0]}")
print(f"   False Positives: {cm[0,1]}")
print(f"   False Negatives: {cm[1,0]}")
print(f"   True Positives: {cm[1,1]}")

# Feature Importance (if using Random Forest)
if isinstance(best_model, RandomForestClassifier):
    print("\n⭐ Top 10 Important Features:")
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-10:][::-1]
    
    # Get feature names
    all_features = ['tfidf_' + str(i) for i in range(X_text.shape[1])]
    all_features.extend(nlp_features_df.columns.tolist())
    
    for i, idx in enumerate(indices):
        if idx < len(all_features):
            feature_name = all_features[idx]
            importance = importances[idx]
            print(f"   {i+1}. {feature_name}: {importance:.4f}")

# Save Models
print("\n💾 Saving models...")
pickle.dump(best_model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(nlp_features_df.columns.tolist(), open("models/feature_names.pkl", "wb"))

print("   ✅ model.pkl")
print("   ✅ vectorizer.pkl")
print("   ✅ scaler.pkl")
print("   ✅ feature_names.pkl")

print("\n✨ Training Complete!")
print("All models saved successfully in /models directory")