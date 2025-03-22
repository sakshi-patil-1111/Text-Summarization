import re
import pdfplumber
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import os

# Download required resources (only needed the first time)
nltk.download("punkt")
nltk.download("stopwords")


def clean_text(text):
    """Removes references, multiple spaces, and special characters from text."""
    text = re.sub(r"\[[0-9]*\]", " ", text)  # Remove references like [1], [2], etc.
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces
    return text


def remove_special_chars(text):
    """Removes special characters and digits, keeping only letters."""
    formatted_text = re.sub("[^a-zA-Z]", " ", text)
    formatted_text = re.sub(r"\s+", " ", formatted_text)  # Remove extra spaces
    return formatted_text

#data extraction 
def extract_text_from_file(input_file):
    """Extracts text from a PDF or TXT file."""
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

    return text


def tokenize_sentences(text):
    """Splits text into sentences."""
    return sent_tokenize(text)


def tokenize_words(text):
    """Splits text into words."""
    return word_tokenize(text)


def remove_stopwords(words):
    """Removes stopwords from a list of words."""
    stop_words = set(stopwords.words("english"))
    return [word for word in words if word.lower() not in stop_words]


def process_file(input_file):
    """Extracts text, cleans it, tokenizes sentences and words, removes stopwords, and saves output."""
    
    # Extract text from file
    text = extract_text_from_file(input_file)
    
    # Clean and format text
    cleaned_text = clean_text(text)
    formatted_text = remove_special_chars(cleaned_text)

    # Tokenization
    sentences = tokenize_sentences(cleaned_text)
    words = tokenize_words(formatted_text)
    filtered_words = remove_stopwords(words)

    # Save tokens to a separate file for each input file
    output_filename = f"tokens_{os.path.splitext(input_file)[0]}.txt"
    with open(output_filename, "w", encoding="utf-8") as out_file:
        out_file.write("\n".join(filtered_words))  

    # Print outputs
    print(f"\nProcessed File: {input_file}")
    print("Cleaned Text (First 500 characters):")
    print(cleaned_text[:500])

    print("\nFormatted Text (First 500 characters):")
    print(formatted_text[:500])

    print("\nFirst 5 Sentences:")
    print(sentences[:5])  

    print("\nFirst 20 Words after Stopword Removal:")
    print(filtered_words[:20])

    print(f"\nTokens have been saved to {output_filename}")


# Example Usage
process_file("file2.pdf")  # Change this to any PDF or TXT file name


if __name__ == "__main__":
    main()
