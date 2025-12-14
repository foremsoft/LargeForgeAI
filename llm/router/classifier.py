"""Query classifier for routing to expert models."""

from dataclasses import dataclass
from typing import Literal

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class ExpertConfig:
    """Configuration for an expert model."""

    name: str
    endpoint: str
    description: str
    keywords: list[str] | None = None


# Default expert configurations
DEFAULT_EXPERTS = [
    ExpertConfig(
        name="code",
        endpoint="http://localhost:8001/generate",
        description="Programming and code-related queries",
        keywords=["code", "python", "javascript", "function", "debug", "error", "programming"],
    ),
    ExpertConfig(
        name="math",
        endpoint="http://localhost:8002/generate",
        description="Mathematics and calculations",
        keywords=["calculate", "math", "equation", "solve", "formula", "number"],
    ),
    ExpertConfig(
        name="writing",
        endpoint="http://localhost:8003/generate",
        description="Creative writing and text generation",
        keywords=["write", "essay", "story", "summarize", "explain", "describe"],
    ),
    ExpertConfig(
        name="general",
        endpoint="http://localhost:8000/generate",
        description="General knowledge and conversation",
        keywords=None,  # Default fallback
    ),
]


class KeywordClassifier:
    """Simple keyword-based classifier for expert routing."""

    def __init__(self, experts: list[ExpertConfig] | None = None):
        self.experts = experts or DEFAULT_EXPERTS
        self._build_keyword_map()

    def _build_keyword_map(self):
        """Build keyword to expert mapping."""
        self.keyword_map: dict[str, str] = {}
        self.default_expert = "general"

        for expert in self.experts:
            if expert.keywords is None:
                self.default_expert = expert.name
            else:
                for keyword in expert.keywords:
                    self.keyword_map[keyword.lower()] = expert.name

    def classify(self, query: str) -> str:
        """Classify query to an expert based on keywords."""
        query_lower = query.lower()
        for keyword, expert_name in self.keyword_map.items():
            if keyword in query_lower:
                return expert_name
        return self.default_expert

    def get_endpoint(self, expert_name: str) -> str | None:
        """Get endpoint URL for an expert."""
        for expert in self.experts:
            if expert.name == expert_name:
                return expert.endpoint
        return None


class NeuralClassifier:
    """Neural network-based classifier for expert routing."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        experts: list[ExpertConfig] | None = None,
        device: str = "auto",
    ):
        self.experts = experts or DEFAULT_EXPERTS
        self.expert_names = [e.name for e in self.experts]
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the classification model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.expert_names),
        )
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def classify(self, query: str) -> str:
        """Classify query using the neural model."""
        if self.model is None:
            self.load_model()

        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_idx = outputs.logits.argmax(dim=-1).item()

        return self.expert_names[predicted_idx]

    def get_endpoint(self, expert_name: str) -> str | None:
        """Get endpoint URL for an expert."""
        for expert in self.experts:
            if expert.name == expert_name:
                return expert.endpoint
        return None

    def train_classifier(self, train_data: list[tuple[str, str]], epochs: int = 3):
        """
        Fine-tune the classifier on labeled data.

        Args:
            train_data: List of (query, expert_name) tuples
            epochs: Number of training epochs
        """
        from torch.utils.data import DataLoader, Dataset as TorchDataset

        if self.model is None:
            self.load_model()

        class ClassificationDataset(TorchDataset):
            def __init__(inner_self, data, tokenizer, label_map):
                inner_self.data = data
                inner_self.tokenizer = tokenizer
                inner_self.label_map = label_map

            def __len__(inner_self):
                return len(inner_self.data)

            def __getitem__(inner_self, idx):
                query, label = inner_self.data[idx]
                encoding = inner_self.tokenizer(
                    query,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors="pt",
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": torch.tensor(inner_self.label_map[label]),
                }

        label_map = {name: idx for idx, name in enumerate(self.expert_names)}
        dataset = ClassificationDataset(train_data, self.tokenizer, label_map)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

        self.model.eval()


class HybridClassifier:
    """Hybrid classifier using keywords with neural fallback."""

    def __init__(
        self,
        experts: list[ExpertConfig] | None = None,
        neural_model: str | None = None,
    ):
        self.keyword_classifier = KeywordClassifier(experts)
        self.neural_classifier = None
        if neural_model:
            self.neural_classifier = NeuralClassifier(neural_model, experts)

    def classify(self, query: str, confidence_threshold: float = 0.8) -> str:
        """
        Classify using keywords first, fall back to neural if uncertain.
        """
        # Try keyword classification
        keyword_result = self.keyword_classifier.classify(query)

        # If we got a non-default result, use it
        if keyword_result != self.keyword_classifier.default_expert:
            return keyword_result

        # Fall back to neural classifier if available
        if self.neural_classifier:
            return self.neural_classifier.classify(query)

        return keyword_result

    def get_endpoint(self, expert_name: str) -> str | None:
        """Get endpoint URL for an expert."""
        return self.keyword_classifier.get_endpoint(expert_name)
