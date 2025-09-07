# src/sentiment_analysis.py

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

# Make sure to download the VADER lexicon if not already
nltk.download('vader_lexicon')

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    """
    Analyze sentiment using VADER.
    Returns: dict with compound, pos, neu, neg scores
    """
    return vader_analyzer.polarity_scores(text)

def analyze_sentiment_textblob(text):
    """
    Analyze sentiment using TextBlob.
    Returns: polarity (-1 to 1) and subjectivity (0 to 1)
    """
    blob = TextBlob(text)
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity
    }

# Example usage
if __name__ == "__main__":
    sample_text = "I am really happy with the outcome of this project, great job everyone!"
    
    vader_result = analyze_sentiment_vader(sample_text)
    textblob_result = analyze_sentiment_textblob(sample_text)

    print("VADER Result:", vader_result)
    print("TextBlob Result:", textblob_result)
