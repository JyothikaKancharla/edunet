from flask import Flask, request, render_template, redirect, url_for, session
import re
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)  # Required for session management


# Build LSTM model
def create_model():
    model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=200),
        SpatialDropout1D(0.2),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create and compile the model
model = create_model()

# Dummy weights for demonstration (use real weights from training in a production app)
model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

# Tokenizer initialization
tokenizer = Tokenizer(num_words=5000)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text
# Main upload page
@app.route("/")
def upload_page():
    return render_template("index1.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Collect user input
            title = request.form.get("title", "")
            company_profile = request.form.get("company_profile", "")
            description = request.form.get("description", "")
            requirements = request.form.get("requirements", "")
            benefits = request.form.get("benefits", "")

            # Combine inputs into one text string
            combined_text = " ".join([title, company_profile, description, requirements, benefits])
            combined_text = preprocess_text(combined_text)

            # Tokenize and pad input
            input_seq = tokenizer.texts_to_sequences([combined_text])
            input_padded = pad_sequences(input_seq, maxlen=200)

            # Get prediction (random results for demonstration)
            prediction = model.predict(input_padded)[0][0]

            # Determine risk level
            risk_level = "Low Risk"
            if prediction > 0.7:
                risk_level = "High Risk"
            elif prediction > 0.4:
                risk_level = "Moderate Risk"

            return render_template("result.html", 
                                   risk_level=risk_level, 
                                   probability=f"{prediction:.2%}")

        except Exception as e:
            return f"An error occurred: {str(e)}"
    return "Invalid request method."

if __name__ == "__main__":
    app.run(debug=True)
