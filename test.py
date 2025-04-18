from flask import Flask, request, jsonify, render_template
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import numpy as np
import cv2
import os
import gdown
import zipfile
from transformers import BartTokenizer, BartForConditionalGeneration
import evaluate

def ensure_model_downloaded(folder_path="./fine_tuned_model", folder_id="1Y4FuXQoGgpYjdeJsskrYI6xmetgYr1sB"):
    if not os.path.exists(folder_path):
        print("Model folder not found. Downloading from Google Drive...")
        gdown.download_folder(id=folder_id, output=folder_path, quiet=False, use_cookies=False)

        

app = Flask(__name__)

# Load fine-tuned model & tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
ensure_model_downloaded()
model = BartForConditionalGeneration.from_pretrained("./fine_tuned_model")

# ----------------------- SUMMARIZATION FUNCTION -----------------------

def generate_summary(text):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=False)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=min(1024, len(text.split())),
        min_length=min(500, len(text.split())//3),
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # return len(text.split())

rouge = evaluate.load("rouge")

def evaluate_summary(predictions, references):
    results = rouge.compute(predictions=predictions, references=references)

    print("ROUGE Scores:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    accurate = 0
    for pred, ref in zip(predictions, references):
        if any(word in ref for word in pred.split()):
            accurate += 1

    accuracy_percent = accurate / len(predictions) * 100
    print(f"Sentence-level ROUGE-1 Overlap Accuracy: {accuracy_percent:.2f}%")


# ----------------------- ROUTES -----------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"summary": "No text provided."}), 400
    summary = generate_summary(text)
    return jsonify({"summary": summary})

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"summary": "No file uploaded."}), 400

    content = ""
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        content = extract_text_from_pdf(file)
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        content = extract_text_from_image(file)
    else:
        return jsonify({"summary": "Unsupported file format."}), 400

    if not content.strip():
        return jsonify({"summary": "No readable content found."}), 400

    summary = generate_summary(content)
    return jsonify({"summary": summary})

# ----------------------- PDF TEXT EXTRACTION -----------------------

def extract_text_from_pdf(file):
    content = ""
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            # Try standard text
            text = page.get_text("text")
            if not text.strip():
                # Fallback to block-based text
                blocks = page.get_text("dict")["blocks"]
                for b in blocks:
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            content += span["text"] + " "
                    content += "\n"
            else:
                content += text + "\n"
    except Exception as e:
        print(f"PDF extraction error: {e}")
    return content

# ----------------------- OCR IMAGE PREPROCESSING -----------------------

def extract_text_from_image(file):
    try:
        image = Image.open(file.stream).convert("RGB")
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR

        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 15, 10
        )
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

        processed_image = Image.fromarray(gray)
        content = pytesseract.image_to_string(processed_image, lang="eng", config="--psm 6")
        return content
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

# ----------------------- MAIN -----------------------

if __name__ == "__main__":
    app.run(debug=True)

