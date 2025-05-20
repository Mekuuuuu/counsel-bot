from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SentimentBERT:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.initialize_model()

    def initialize_model(self):
        # Initialize BERT model for sentiment analysis
        model_name = os.getenv("SENTIMENT_MODEL", "Mekuu/BERT-A-Sentiment")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            token=os.getenv("HUGGINGFACE_TOKEN"),
            torch_dtype=getattr(torch, os.getenv("TORCH_DTYPE", "float16"))
        ).to(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

    def predict_sentiment(self, text: str) -> dict:
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()

        # Map prediction to label
        sentiment_map = {
            0: "sadness",
            1: "anger",
            2: "love",
            3: "surprise",
            4: "fear",
            5: "joy"
        }

        # Get probabilities for each class
        probs = {
            sentiment_map[i]: round(prob.item() * 100, 2)
            for i, prob in enumerate(probabilities[0])
        }

        return {
            "sentiment": sentiment_map[predicted_class],
            "probabilities": probs
        }

# Create singleton instance
sentiment_bert = SentimentBERT()

def predict_sentiment(text: str) -> dict:
    return sentiment_bert.predict_sentiment(text) 