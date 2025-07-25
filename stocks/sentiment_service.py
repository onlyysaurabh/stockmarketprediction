import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "ProsusAI/finbert" # Using the standard FinBERT model
DEVICE = 0 if torch.cuda.is_available() else -1 # Use GPU if available, otherwise CPU

# Global variables to hold the model and tokenizer (lazy loaded)
tokenizer = None
model = None
sentiment_pipeline = None

def _load_model():
    """Loads the FinBERT model and tokenizer if not already loaded."""
    global tokenizer, model, sentiment_pipeline
    if sentiment_pipeline is None:
        try:
            logger.info(f"Loading FinBERT model '{MODEL_NAME}'...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            # device=DEVICE uses GPU if available
            sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=DEVICE)
            logger.info("FinBERT model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model '{MODEL_NAME}': {e}", exc_info=True)
            # Set to False to prevent repeated load attempts on error
            sentiment_pipeline = False 
    
    # Return True if loaded successfully or already loaded, False otherwise
    return sentiment_pipeline is not False and sentiment_pipeline is not None


def analyze_sentiment(text: str) -> Optional[Dict[str, float]]:
    """
    Analyzes the sentiment of a given text using FinBERT.

    Args:
        text: The text content (e.g., news headline + summary) to analyze.

    Returns:
        A dictionary containing sentiment scores {'positive': float, 'negative': float, 'neutral': float},
        or None if analysis fails or the model couldn't be loaded.
    """
    if not text:
        logger.warning("Received empty text for sentiment analysis.")
        return None

    if not _load_model():
        logger.error("Sentiment analysis cannot proceed because the model is not loaded.")
        return None

    try:
        # The pipeline returns a list of results, even for single input
        # [{'label': 'neutral', 'score': 0.8...}]
        results = sentiment_pipeline(text, return_all_scores=True)
        
        if not results or not isinstance(results, list) or not results[0]:
             logger.warning(f"Sentiment analysis returned unexpected result format for text: {text[:100]}...")
             return None

        # FinBERT returns scores for positive, negative, neutral
        scores = {item['label']: item['score'] for item in results[0]}
        
        # Ensure all expected labels are present
        final_scores = {
            'positive': scores.get('positive', 0.0),
            'negative': scores.get('negative', 0.0),
            'neutral': scores.get('neutral', 0.0)
        }
        
        logger.debug(f"Sentiment analysis result for '{text[:50]}...': {final_scores}")
        return final_scores

    except Exception as e:
        logger.error(f"Error during sentiment analysis for text '{text[:100]}...': {e}", exc_info=True)
        return None

# Example Usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    test_texts = [
        "Stock prices surged after the positive earnings report.",
        "The company announced unexpected losses, causing shares to plummet.",
        "The market remained flat today with little movement.",
        "Regulatory concerns might impact future growth.", # More neutral/negative
    ]

    print("Attempting sentiment analysis (will download model on first run)...")
    for text in test_texts:
        sentiment = analyze_sentiment(text)
        print(f"\nText: {text}")
        if sentiment:
            print(f"Sentiment: Positive={sentiment['positive']:.4f}, Negative={sentiment['negative']:.4f}, Neutral={sentiment['neutral']:.4f}")
        else:
            print("Sentiment analysis failed.")
