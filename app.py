# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import os
# import re
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# from collections import defaultdict
# import heapq
# import pytesseract
# from PIL import Image
# import pdfplumber
# from data_extraction1 import extract_and_tokenize
# app = Flask(__name__)
# CORS(app)

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True) 
# proxy = "http://edcguest:edcguest@172.31.102.29:3128"
# nltk.set_proxy(proxy)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# @app.route('/')
# def index():
#     return render_template('index.html')


# def clean_text(text):
#     text = re.sub(r'\[[0-9]*\]', ' ', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text

# def remove_special_chars(text):
#     formatted_text = re.sub('[^a-zA-Z]', ' ', text)
#     formatted_text = re.sub(r'\s+', ' ', formatted_text)
#     return formatted_text

# def tokenize_sentences(text):
#     return sent_tokenize(text)

# def tokenize_words(text):
#     return word_tokenize(text)

# def remove_stopwords(words):
#     stop_words = set(stopwords.words('english'))
#     return [word for word in words if word.lower() not in stop_words]

# def compute_word_frequencies(words):
#     word_frequencies = defaultdict(int)
#     for word in words:
#         word_frequencies[word] += 1
#     max_frequency = max(word_frequencies.values(), default=1)
#     for word in word_frequencies:
#         word_frequencies[word] /= max_frequency
#     return word_frequencies

# def score_sentences(sentences, word_frequencies):
#     sentence_scores = defaultdict(int)
#     for sent in sentences:
#         for word in word_tokenize(sent.lower()):
#             if word in word_frequencies and len(sent.split(' ')) < 30:
#                 sentence_scores[sent] += word_frequencies[word]
#     return sentence_scores

# def get_summary(sentence_scores, n=7):
#     summary_sentences = heapq.nlargest(n, sentence_scores, key=sentence_scores.get)
#     summary = ' '.join(summary_sentences)
#     return summary

# @app.route('/summarize', methods=['POST'])
# def summarize_text():
#     data = request.get_json()
#     if not data or 'text' not in data:
#         return jsonify({'error': 'No text provided'}), 400

#     text = data['text']
#     cleaned_text = clean_text(text)
#     formatted_text = remove_special_chars(cleaned_text)
#     sentences = tokenize_sentences(cleaned_text)
#     words = tokenize_words(formatted_text)
#     filtered_words = remove_stopwords(words)
#     word_frequencies = compute_word_frequencies(filtered_words)
#     sentence_scores = score_sentences(sentences, word_frequencies)
#     summary = get_summary(sentence_scores)

#     return jsonify({'summary': summary})

# def extract_text_from_image(image_path):
#     try:
#         img = Image.open(image_path)
#         text = pytesseract.image_to_string(img)
#         return text.strip()
#     except Exception as e:
#         return f"Error extracting text: {str(e)}"

# def extract_text_from_pdf(pdf_path):
#     text = ""
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 extracted_text = page.extract_text()
#                 if extracted_text:
#                     text += extracted_text + " "
#     except Exception as e:
#         return f"Error reading PDF: {str(e)}"
#     return text.strip()

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400

#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)

#     if file.filename.endswith('.pdf') or file.filename.endswith('.txt'):
#         extracted_text = extract_and_tokenize(file_path)  # Now returns full text
#     elif file.filename.endswith(('.png', '.jpg', '.jpeg')):
#         extracted_text = extract_text_from_image(file_path)
#     else:
#         return jsonify({'error': 'Unsupported file format'}), 400

#     summary = summarize_text_from_input(extracted_text)  # Summarize full text
#     return jsonify({'summary': summary})


# def summarize_text_from_input(text):
#     sentences = tokenize_sentences(text)
#     words = tokenize_words(text)
#     filtered_words = remove_stopwords(words)
#     word_frequencies = compute_word_frequencies(filtered_words)
#     sentence_scores = score_sentences(sentences, word_frequencies)
#     return get_summary(sentence_scores)

# if __name__ == '__main__':
#     app.run(debug=True)
