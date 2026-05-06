import os
import re
import glob
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, collection_name="laws"):
        qdrant_url = os.environ.get("QDRANT_URL", ":memory:")
        self.client = QdrantClient(qdrant_url)
        self.collection_name = collection_name
        
        embed_model_id = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.embed_model = SentenceTransformer(embed_model_id)
        # Fixed: get_sentence_embedding_dimension is deprecated
        self.vector_size = self.embed_model.get_embedding_dimension()
        
        # Create collection if it doesn't exist
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def chunk_legal_document(self, filepath: str) -> list[dict]:
        """Split a legal markdown file by clause headers (## or ###).
        Prepend the Act title to each chunk for context."""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract act title from H1
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else os.path.basename(filepath)
        
        # Split by clause headers
        sections = re.split(r'\n(?=##\s|###\s)', content)
        
        chunks = []
        for i, section in enumerate(sections):
            section = section.strip()
            if len(section) > 50:  # Skip tiny fragments
                chunks.append({
                    'id': i, # This will be overwritten by index_all
                    'text': f"[{title}] {section}",
                    'metadata': {'source': filepath, 'act': title}
                })
        return chunks

    def index_documents(self, documents: list[dict]):
        """
        documents: list of dicts with 'id' (int), 'text' (str), 'metadata' (dict)
        """
        if not documents:
            return

        texts = [doc['text'] for doc in documents]
        embeddings = self.embed_model.encode(texts)
        
        points = [
            PointStruct(
                id=doc['id'],
                vector=embedding.tolist(),
                payload={"text": doc['text'], **doc.get('metadata', {})}
            )
            for doc, embedding in zip(documents, embeddings)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Indexed {len(documents)} points into '{self.collection_name}'.")

    def index_all_mock_data(self, data_dir="mock_data"):
        all_chunks = []
        global_id = 0
        for filepath in glob.glob(os.path.join(data_dir, "*.md")):
            chunks = self.chunk_legal_document(filepath)
            for chunk in chunks:
                chunk['id'] = global_id
                all_chunks.append(chunk)
                global_id += 1
        self.index_documents(all_chunks)

    def search(self, query: str, top_k: int = 3):
        query_vector = self.embed_model.encode(query).tolist()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )
        return [hit.payload for hit in results.points]
