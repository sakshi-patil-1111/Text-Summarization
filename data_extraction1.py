import pdfplumber
import nltk
import re
import string


nltk.download("punkt")


def extract_and_tokenize(input_file, output_file="tokens_output.txt"):
    ext = input_file.rsplit(".", 1)[-1].lower()
    text = ""

    if ext == "txt":
        with open(input_file, "r", encoding="utf-8") as file:
            text = file.read()
    elif ext == "pdf":
        with pdfplumber.open(input_file) as pdf:
            for page in pdf.pages:
                extracted_page_text = page.extract_text()
                if extracted_page_text:
                    text += extracted_page_text + " "
    else:
        raise ValueError("Unsupported file format. Only TXT and PDF are allowed.")

    text = re.sub(r"\s+", " ", text)  
    
    tokens = nltk.word_tokenize(text) 
    tokens = [word for word in tokens if word not in string.punctuation]  # Remove punctuation

    with open(output_file, "w", encoding="utf-8") as out_file:
        out_file.write("\n".join(tokens))  

    return text 
