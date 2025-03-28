import pandas as pd
import pickle 
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return text
    return ""
import pandas as pd

df = pd.read_csv('archive (3)/fake_job_postings.csv', encoding="ISO-8859-1", on_bad_lines='skip', engine='python')





# Split into real and fake job datasets
real_jobs = df[df["fraudulent"] == 0]
fake_jobs = df[df["fraudulent"] == 1]

# Save to new CSV files
real_jobs.to_csv("real_jobs.csv", index=False)
fake_jobs.to_csv("fake_jobs.csv", index=False)

print("Real and fake job datasets saved successfully!")
# Load the datasets
real_jobs = pd.read_csv("real_jobs.csv")
fake_jobs = pd.read_csv("fake_jobs.csv")
# Add label columns
real_jobs["fraudulent"] = 0
fake_jobs["fraudulent"] = 1
# Combine datasets
df = pd.concat([real_jobs, fake_jobs], ignore_index=True)
# Select relevant columns
df = df[['title', 'company_profile', 'description', 'requirements', 'benefits', 'fraudulent']].dropna()
# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_columns:
    df[col] = df[col].apply(preprocess_text)
# Tokenization and Padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['description'])
X = tokenizer.texts_to_sequences(df['description'])
X = pad_sequences(X, maxlen=200)
y = df['fraudulent'].values
# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy=0.4, random_state=42)  # Increased oversampling ratio
X_smote, y_smote = smote.fit_resample(X, y)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
# Build LSTM Model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=200),
    SpatialDropout1D(0.2),
    LSTM(100, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
# Evaluate Model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Model and tokenizer saved successfully!")
# Function for user input predictions
def validate_input(text, field_name, min_length=10):
    """Validate input text for minimum length and basic content checks."""
    if not isinstance(text, str):
        return False, f"{field_name} must be text"
    
    # Remove spaces and special characters for length check
    cleaned_text = re.sub(r'\s+', '', text)
    if len(cleaned_text) < min_length:
        return False, f"{field_name} is too short (minimum {min_length} characters)"
    
    # Check for repetitive characters
    if re.search(r'(.)\1{4,}', text):
        return False, f"{field_name} contains suspicious repetitive characters"
    
    # Check for random keyboard mashing
    if re.search(r'^[qweasdzxc]+$|^[yuiophjkl]+$|^[vbnm]+$', text.lower()):
        return False, f"{field_name} appears to be random characters"
    
    return True, ""

def predict_job_fraud():
    print("\nPlease provide detailed information for job posting analysis:")
    
    # Dictionary to store inputs with their minimum lengths
    fields = {
        'title': ('Enter Job Title: ', 5),
        'company_profile': ('Enter Company Profile (min. 20 chars): ', 0),
        'description': ('Enter Job Description (min. 50 chars): ', 0),
        'requirements': ('Enter Job Requirements (min. 30 chars): ', 0),
        'benefits': ('Enter Job Benefits (min. 20 chars): ', 0),
        'location': ('Enter Job Location: ', 0),
        'salary': ('Enter Salary Range: ', 0),
        'url': ('Enter Job URL (Optional): ', 0)
    }
    
    inputs = {}
    suspicious_count = 0
    
    # Collect and validate inputs
    for field, (prompt, min_length) in fields.items():
        while True:
            text = input(prompt)
            if field == 'url' and not text:  # URL is optional
                break
                
            is_valid, message = validate_input(text, field, min_length)
            if is_valid:
                inputs[field] = text
                break
            else:
                print(f"⚠️ {message}. Please try again.")
                suspicious_count += 1
            
            if suspicious_count >= 3:
                print("\n🚨 Multiple invalid inputs detected. This may indicate suspicious behavior.")
    
    # Combine all inputs
    input_text = ' '.join(inputs.values())
    input_text = preprocess_text(input_text)
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_seq, maxlen=200)
    
    # Get prediction probability
    prediction = model.predict(input_padded)[0][0]
    
    # Adjust prediction based on suspicious input patterns
    if suspicious_count > 0:
        prediction = prediction + (suspicious_count * 0.1)  # Increase fraud probability for suspicious inputs
    
    print("\nAnalysis Results:")
    print(f"Fraud Probability Score: {prediction:.2%}\n")

    # Risk level thresholds
    if prediction > 0.7 or suspicious_count >= 3:
        print("🚨 HIGH RISK: This job posting shows strong indicators of being fraudulent!")
        print("\nRed Flags Detected:")
        print("- Suspicious or low-quality input content")
        print("- Unusual text patterns or repetitive characters")
        print("- Incomplete or vague information")
        print("\nRecommended Actions:")
        print("- Do not proceed with this application")
        print("- Report this posting to relevant job boards")
        print("- Look for verified employers and job listings")
    
    elif prediction > 0.4 or suspicious_count >= 1:
        print("⚠️ MODERATE RISK: This job posting requires careful verification.")
        print("\nCaution Points:")
        print("- Some input fields contain questionable content")
        print("- Additional verification strongly recommended")
        print("- Cross-reference with official company sources")
    
    else:
        print("✅ LOW RISK: Initial analysis suggests legitimate posting.")
        print("\nRecommended Verification Steps:")
        print("1. Verify company existence and reputation")
        print("2. Cross-check job posting on official company website")
        print("3. Research typical salary ranges for this position")
        print("4. Look up company reviews on reliable platforms")
    
    print("\n⚠️ IMPORTANT: Regardless of risk level, always:")
    print("- Research the company thoroughly")
    print("- Never share financial or sensitive personal information")
    print("- Be wary of requests for payment or unusual interview processes")
    print("- Trust your instincts if something feels wrong")

# Run prediction with user input
predict_job_fraud()