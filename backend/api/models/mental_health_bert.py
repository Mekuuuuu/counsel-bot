from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MentalHealthBERT:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.initialize_model()

    def initialize_model(self):
        # Initialize BERT model for mental health classification
        model_name = os.getenv("MENTAL_HEALTH_MODEL", "Mekuu/BERT-B-Mental-State")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            token=os.getenv("HUGGINGFACE_TOKEN"),
            torch_dtype=getattr(torch, os.getenv("TORCH_DTYPE", "float16"))
        ).to(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

    def classify_mental_health(self, text: str) -> dict:
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
        classification_map = {
            0: "normal",
            1: "depression",
            2: "suicidal",
            3: "anxiety",
            4: "bipolar",
            5: "stress",
            6: "personality disorder"
        }

        # Get probabilities for each class
        probs = {
            classification_map[i]: round(prob.item() * 100, 2) #sentiment_map[i]: round(prob.item() * 100, 2)
            for i, prob in enumerate(probabilities[0])
        }

        return {
            "classification": classification_map[predicted_class],
            "probabilities": probs
        }

# Create singleton instance
mental_health_bert = MentalHealthBERT()

def classify_mental_health(text: str) -> dict:
    return mental_health_bert.classify_mental_health(text) 