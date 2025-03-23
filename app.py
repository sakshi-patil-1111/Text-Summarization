import re
import pdfplumber
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import os
import heapq


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


def clean_text(text):
    text = re.sub(r'\[[0-9]*\]', ' ', text) 
    text = re.sub(r'\s+', ' ', text) 
    return text

def remove_special_chars(text):
    formatted_text = re.sub('[^a-zA-Z]', ' ', text)  
    formatted_text = re.sub(r'\s+', ' ', formatted_text)
    return formatted_text


def tokenize_sentences(text):
    return sent_tokenize(text)


def tokenize_words(text):
    return word_tokenize(text)


def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return filtered_words


def compute_word_frequencies(words):
    word_frequencies = defaultdict(int)
    for word in words:
        word_frequencies[word] += 1
    max_frequency = max(word_frequencies.values())

    for word in word_frequencies:
        word_frequencies[word] /= max_frequency  
    return word_frequencies


def score_sentences(sentences, word_frequencies):
    sentence_scores = defaultdict(int)

    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                if len(sent.split(' ')) < 30:  
                    sentence_scores[sent] += word_frequencies[word]
    return sentence_scores



def get_summary(sentence_scores, n=7):
    summary_sentences = heapq.nlargest(n, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary


def main():
    
    text = """Artificial Intelligence (AI) is revolutionizing industries across the globe, transforming 
    the way people work, communicate, and interact with technology. AI-powered systems are now used in healthcare
    to diagnose diseases, in finance to detect fraud, and in autonomous vehicles to enhance road safety. Despite 
    these advancements, AI also raises ethical concerns regarding data privacy, job displacement, and decision-making
    biases.
    
    One of the most significant benefits of AI is its ability to process large amounts of data efficiently. In the medical field, 
    AI-driven algorithms can analyze patient records to detect diseases at an early stage, leading to better treatment outcomes. 
    Similarly, AI is transforming the financial sector by identifying fraudulent transactions in real-time, reducing financial 
    losses for businesses and individuals.
    
    However, the widespread adoption of AI has led to concerns about job automation. Many routine tasks, such as customer 
    service and manufacturing operations, are increasingly being performed by AI-driven robots, leading to workforce displacement. 
    While new job opportunities are emerging in AI development and maintenance, there is an urgent need for reskilling and upskilling 
    programs to help workers transition to new roles.
    
    Another critical challenge is the ethical implications of AI decision-making. Machine learning models are trained on historical data,
    which can contain biases. If not carefully managed, AI systems may reinforce these biases, leading to unfair treatment in areas such as
    hiring and law enforcement. Ensuring transparency and accountability in AI algorithms is essential to mitigate these risks.
    
    In conclusion, while AI offers immense potential to improve efficiency and decision-making across various sectors, 
    it is crucial to address its ethical and societal challenges. Governments, businesses, and researchers must work together
    to develop regulations and policies that ensure AI is used responsibly and beneficially for all."""

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
    
    word_frequencies = compute_word_frequencies(filtered_words)
    print(dict(list(word_frequencies.items())[:20]))
    
    sentence_scores = score_sentences(sentences, word_frequencies)
    print(dict(list(sentence_scores.items())[:5]))
    
    summary = get_summary(sentence_scores)
    print("\nSummary:")
    print(summary)
