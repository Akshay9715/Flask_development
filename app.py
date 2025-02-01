from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from transformers import pipeline

pipe = pipeline("text2text-generation", model="akshay9125/Transcript_Summerizer")
app = Flask(__name__)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("akshay9125/Transcript_Summerizer")
model = AutoModelForSeq2SeqLM.from_pretrained("akshay9125/Transcript_Summerizer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Function to generate summary
def generate_summary(text, max_length=150):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary




@app.route('/')
def index():
    return render_template('index.html' )



@app.route('/summarize', methods=['POST'])
def summarize():
    # Get data from the request
    data = request.get_json()

    # Ensure the request has 'transcript' key
    if "transcript" not in data:
        return jsonify({"error": "Please provide a 'transcript' key in the JSON payload."}), 400

    transcript = data['transcript']

    # Generate the summary
    summary = generate_summary(transcript)

    return jsonify({"summary": summary})




if __name__ == '__main__':
    app.run(debug=True)




