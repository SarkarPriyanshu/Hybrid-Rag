import torch
from transformers import AutoTokenizer, AutoModel

class BgeM3Embeddings:
    def __init__(self, model_name: str = "BAAI/bge-m3", device=None):
        # Choose device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with safetensors if available
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True  # Use the safer format
        )

        self.model.to(self.device)
        self.model.eval()

    def _embed(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get embeddings, mean-pool, then normalize
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def embed_documents(self, texts):
        return self._embed(texts).tolist()

    def embed_query(self, text):
        return self._embed([text])[0]
