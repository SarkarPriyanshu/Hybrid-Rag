import torch
from transformers import AutoTokenizer, AutoModel

class MiniLMEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device=None):
        # Choose device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model (use safetensors if available for speed & safety)
        self.model = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True
        )

        self.model.to(self.device)
        self.model.eval()

    def _embed(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,  # MiniLM has smaller context window
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean-pool last hidden state to get embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def embed_documents(self, texts):
        return self._embed(texts).tolist()

    def embed_query(self, text):
        return self._embed([text])[0]
