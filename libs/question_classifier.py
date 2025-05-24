import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import os

class QuestionClassifier:
    def __init__(self, classifier_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer, self.label_encoder = self._load_qeustion_classifier(classifier_path)

    def _load_qeustion_classifier(self, classifier_path):
        # Define model path (update this if different)
        model_path = classifier_path

        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load Model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()  # Set to evaluation mode

        # Load Model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()  # Set to evaluation mode

        # 🚀 Move model to GPU if available
        model.to(self.device)

        # Load Label Encoder
        label_encoder = joblib.load("/home/dice/myothiha/table_qa/libs/question_classifiers/Fine-Tuned-RoBERTa/label_encoder.pkl")
        return (model, tokenizer, label_encoder)

    # 🔥 Predict Function (Returns Class Labels)
    def __call__(self, texts):
        """Predict labels for input text(s) and return class names."""
        if isinstance(texts, str):
            texts = [texts]  # Convert single text to list

        # Tokenize inputs & move to correct device
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Model Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_classes = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        # 🔥 Convert Predicted Labels to Class Names
        return self.label_encoder.inverse_transform(predicted_classes)  # Returns class names