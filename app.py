
from flask import Flask, request, render_template
from flask_cors import CORS
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
CORS(app)


# Load the model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Store conversation history (limit length to avoid overflow)
conversation_history = []


# Routes
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

"""
@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    data = json.loads(request.get_data(as_text=True))
    input_text = data.get('prompt', '').strip()
    print(f"[User]: {input_text}")

    # Maintain last 6 exchanges to prevent history overload
    short_history = conversation_history[-6:]

    # Compose context for BlenderBot
    history_text = "\n".join(short_history)
    if history_text:
        combined_input = history_text + "\n" + input_text
    else:
        combined_input = input_text

    # Encode and generate response
    inputs = tokenizer([combined_input], return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=200,              # prevent run-on
        no_repeat_ngram_size=3,      # reduce repeated phrases
        do_sample=True,              # add randomness
        top_k=50, top_p=0.95,        # nucleus sampling
        temperature=0.7               # lower temp -> more sensible
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"[Bot]: {response}")

    # Update history
    conversation_history.append(input_text)
    conversation_history.append(response)

    return response
"""

if __name__ == '__main__':
    app.run()
