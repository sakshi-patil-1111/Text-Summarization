import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


def clean_text(text):
    text = re.sub(r'\[[0-9]*\]', ' ', text)  # Remove references
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    return text

def remove_special_chars(text):
    formatted_text = re.sub('[^a-zA-Z]', ' ', text)  # Remove special characters and digits
    formatted_text = re.sub(r'\s+', ' ', formatted_text) # Remove extra spaces
    return formatted_text


def tokenize_sentences(text):
    return sent_tokenize(text)


def tokenize_words(text):
    return word_tokenize(text)


def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return filtered_words

def main():
    text = """Natural Language Processing (NLP) is a field of artificial intelligence that enables computers to understand, 
    interpret, and generate human language. NLP techniques are used in applications such as chatbots, sentiment analysis, 
    machine translation, and text summarization. One of the key challenges in NLP is understanding context, ambiguity, and 
    linguistic nuances. Various algorithms, including rule-based methods and deep learning models, are employed to process 
    textual data efficiently. NLP continues to evolve with advancements in AI, making human-computer interaction more 
    seamless and intelligent.

    Text summarization is an important application of NLP that helps condense large amounts of textual information into 
    shorter, meaningful summaries. It is widely used in news aggregation, research paper summarization, and document 
    processing. Summarization techniques are broadly categorized into extractive and abstractive methods. Extractive 
    summarization selects key sentences from the original text, while abstractive summarization generates new sentences 
    that capture the main ideas.

    Traditional text summarization methods rely on statistical techniques such as TF-IDF (Term Frequency-Inverse Document 
    Frequency) and sentence ranking algorithms like TextRank. Modern approaches leverage deep learning and transformers, 
    including models like BERT and GPT, which understand context and generate high-quality summaries. The effectiveness 
    of summarization depends on the quality of the input text, preprocessing techniques, and the chosen model.

    As the amount of digital information grows, the demand for automated summarization tools increases. Businesses, 
    researchers, and content creators benefit from AI-powered summarization to extract essential information quickly 
    and efficiently. NLP advancements will continue to improve summarization models, making them more accurate, 
    context-aware, and widely applicable."""
    
    
    cleaned_text = clean_text(text)
    formatted_article_text = remove_special_chars(cleaned_text)

    print("Cleaned Text (First 500 characters):")
    print(cleaned_text[:500]) 
    print("\nFormatted Article Text (First 500 characters):")
    print(formatted_article_text[:500])

    sentences = tokenize_sentences(cleaned_text)
    words = tokenize_words(formatted_article_text)

    filtered_words = remove_stopwords(words)

    print("\nFirst 5 Sentences:")
    print(sentences[:5])  
    
    print("\nFirst 20 Words after Stopword Removal:")
    print(filtered_words[:20]) 


if __name__ == "__main__":
    main()