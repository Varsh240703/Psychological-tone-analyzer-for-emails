# src/formality_score.py
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download NLTK resources if not already available
nltk.download('punkt')
nltk.download('stopwords')

def calculate_formality_score(text):
    # Tokenization
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Counts
    avg_sentence_length = sum(len(word_tokenize(s)) for s in sentences) / len(sentences)
    stop_words = set(stopwords.words('english'))
    stopword_ratio = sum(1 for w in words if w.lower() in stop_words) / len(words)
    uppercase_count = sum(1 for w in words if w.isupper())
    uppercase_ratio = uppercase_count / len(words)

    # Use of contractions (e.g., "don't", "I'm") is more informal
    contractions = re.findall(r"\b(?:[A-Za-z]+\'[A-Za-z]+)\b", text)
    contraction_ratio = len(contractions) / len(words)

    # Use of slang or emojis — you can expand this with a slang dictionary
    emoji_count = len(re.findall(r'[^\w\s,]', text))
    emoji_ratio = emoji_count / len(words)

    # Heuristic formality score
    score = (
        (avg_sentence_length / 20) * 0.4 +         # Longer sentences → more formal
        (1 - contraction_ratio) * 0.2 +            # Fewer contractions → more formal
        (1 - emoji_ratio) * 0.1 +                  # Fewer emojis → more formal
        (1 - uppercase_ratio) * 0.1 +              # Less shouting → more formal
        (stopword_ratio) * 0.2                     # Natural flow of words → more formal
    )

    return round(score * 100, 2)  # Scale to 0–100

if __name__ == "__main__":
    sample_text = "Dear Sir, I hope this message finds you well. I would like to schedule a meeting at your convenience."
    score = calculate_formality_score(sample_text)
    print(f"Formality Score: {score}/100")
