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

DRIVE_FILE_ID = "https://drive.google.com/drive/folders/1Y4FuXQoGgpYjdeJsskrYI6xmetgYr1sB?usp=sharing"



def download_model_from_drive(file_id, output_dir="fine_tuned_model"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        zip_path = "model.zip"
        url = f"https://drive.google.com/uc?id={file_id}"
        print("Downloading model from Google Drive...")
        gdown.download(url, zip_path, quiet=False)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(zip_path)
    else:
        return
        
        

app = Flask(__name__)

# Load fine-tuned model & tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("fine_tuned_model")

# ----------------------- SUMMARIZATION FUNCTION -----------------------

def generate_summary(text):
    download_model_from_drive(DRIVE_FILE_ID)
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

