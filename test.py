from flask import Flask, request, jsonify, render_template
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import numpy as np
import cv2
import os
import gdown
import csv
from io import StringIO
import evaluate

os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.100.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.100.14:3128"

app = Flask(__name__)

def ensure_model_downloaded(folder_path="./fine_tuned_model", folder_id="1Y4FuXQoGgpYjdeJsskrYI6xmetgYr1sB"):
    if not os.path.exists(folder_path):
        print("Downloading model...")
        gdown.download_folder(id=folder_id, output=folder_path, quiet=False, use_cookies=False)

print("Loading model...")
tokenizer = model = None
try:
    ensure_model_downloaded()
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("./fine_tuned_model")
    print("Model loaded.")
except Exception as e:
    print(f"Model load failed: {e}")

try:
    rouge = evaluate.load("rouge")
    print("ROUGE metric loaded.")
except Exception as e:
    print(f" Failed to load ROUGE metric: {e}")
    rouge = None

def generate_summary(text):
    if not model or not tokenizer:
        return "Model not loaded."
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def extract_text_from_pdf(file):
    content = ""
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text = page.get_text("text")
            if not text.strip():
                blocks = page.get_text("dict")["blocks"]
                for b in blocks:
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            content += span["text"] + " "
                    content += "\n"
            else:
                content += text + "\n"
    except Exception as e:
        print(f"PDF extract error: {e}")
    return content

def extract_text_from_image(file):
    try:
        image = Image.open(file.stream).convert("RGB")
        open_cv_image = np.array(image)[:, :, ::-1].copy()
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 15, 10)
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        processed_image = Image.fromarray(gray)
        content = pytesseract.image_to_string(processed_image, lang="eng", config="--psm 6")
        return content
    except Exception as e:
        print(f"OCR error: {e}")
        return ""


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
        return jsonify({"text":"","error": "No file uploaded."}), 400


    filename = file.filename.lower()
    content = ""

    if filename.endswith(".pdf"):
        content = extract_text_from_pdf(file)
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        content = extract_text_from_image(file)
    else:
        return jsonify({"text":"","error": "Unsupported file format."}), 400

    if not content.strip():
        return jsonify({"text": "", "error": "No readable content found."}), 400

    return jsonify({"text": content})

@app.route("/evaluate-summary", methods=["POST"])
def evaluate_summary_route():
    file = request.files.get("file")
    if not file:
        return jsonify({"message": "No file uploaded."}), 400

    try:
        content = file.stream.read().decode("utf-8-sig")
        csv_file = StringIO(content)
        reader = csv.DictReader(csv_file)

        documents = []
        references = []

        for row in reader:
            doc = row.get("document", "").strip()
            summary = row.get("summary", "").strip()
            if doc and summary:
                documents.append(doc)
                references.append(summary)

        if not documents or not references:
            return jsonify({"message": "No valid rows found in CSV."}), 400

        predictions = []

        for doc in documents:
            inputs = tokenizer([doc], max_length=1024, return_tensors="pt", truncation=True).to(model.device)
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=256,
                min_length=10,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
            predicted_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            predictions.append(predicted_summary.strip())

        # Compute ROUGE
        results = rouge.compute(predictions=predictions, references=references)

        formatted_results = {}
        for k, v in results.items():
            try:
                formatted_results[k] = f"{v * 100:.2f}%"
            except:
                formatted_results[k] = "Error"

        # More accurate sentence-level word overlap logic
        accurate = 0
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            if pred_words & ref_words:
                accurate += 1

        accuracy_percent = accurate / len(predictions) * 100
        formatted_results["sentence_overlap_accuracy"] = f"{accuracy_percent:.2f}%"

        return jsonify({
            "scores": formatted_results,
            
        })

    except Exception as e:
        print(f" Evaluation error: {e}")
        return jsonify({"message": "Error processing file."}), 500


if __name__ == "__main__":
    print(" Flask app starting...")
    app.run(debug=True)
