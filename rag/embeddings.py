class Embeddings:
    def __init__(self, model_name, cache_dir):
        """Remember the semantic model name and keep downloaded weights in the data folder."""
        self.name, self.cache_dir, self.model = model_name, str(cache_dir), None

    def encode(self, texts):
        """Use Sentence Transformers to produce normalized multilingual semantic vectors."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.name, cache_folder=self.cache_dir)
            self.model.max_seq_length = 512
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
