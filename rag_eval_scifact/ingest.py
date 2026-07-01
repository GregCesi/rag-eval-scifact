"""Ingestion du corpus SciFact dans ChromaDB.

Charge les 5183 docs depuis corpus.jsonl (fichiers bruts BEIR),
les embedde avec all-MiniLM-L6-v2 (max_seq_length=256),
et les indexe dans une collection ChromaDB fraîche en espace cosinus.

Calcule et stocke :
- token_count par doc (AutoTokenizer, add_special_tokens=True)
- dataset_hash (SHA-256 du fichier corpus.jsonl brut)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# --- Constantes ---

CORPUS_PATH = Path("data/scifact/corpus.jsonl")
CHROMA_DIR = Path("chroma_data")
COLLECTION_NAME = "scifact_v1"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256
EXPECTED_COUNT = 5183


def compute_dataset_hash(corpus_path: Path) -> str:
    """SHA-256 du fichier corpus.jsonl brut (source de vérité)."""
    h = hashlib.sha256()
    with open(corpus_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def load_corpus(corpus_path: Path) -> list[dict]:
    """Charge le corpus BEIR SciFact depuis corpus.jsonl.

    Retourne une liste de dicts avec _id, title, text.
    """
    docs = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            docs.append(doc)
    return docs


def ingest() -> None:
    """Pipeline d'ingestion complet."""
    print(f"Chargement du corpus depuis {CORPUS_PATH}...")
    docs = load_corpus(CORPUS_PATH)
    assert len(docs) == EXPECTED_COUNT, f"Attendu {EXPECTED_COUNT} docs, trouvé {len(docs)}"
    print(f"  {len(docs)} documents chargés.")

    # Concaténation title + " " + text
    texts = [doc["title"] + " " + doc["text"] for doc in docs]
    doc_ids = [doc["_id"] for doc in docs]
    titles = [doc["title"] for doc in docs]

    # Token count avec AutoTokenizer (add_special_tokens=True, identique à l'inspection étape 2)
    print("Calcul des token_count...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    token_counts = [
        len(tokenizer.encode(text, add_special_tokens=True))
        for text in texts
    ]
    truncated = sum(1 for tc in token_counts if tc > MAX_SEQ_LENGTH)
    print(f"  {truncated}/{len(docs)} docs dépassent {MAX_SEQ_LENGTH} tokens (troncature acceptée).")

    # Embedding avec SentenceTransformer
    print("Embedding des documents avec MiniLM...")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings_list = [emb.tolist() for emb in embeddings]
    print(f"  {len(embeddings_list)} embeddings calculés (dim={len(embeddings_list[0])}).")

    # Dataset hash
    dataset_hash = compute_dataset_hash(CORPUS_PATH)
    print(f"  dataset_hash = {dataset_hash[:30]}...")

    # ChromaDB — collection fraîche en cosinus
    print("Indexation dans ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Supprimer l'ancienne collection si elle existe (hnsw:space n'est lu qu'à la création)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Collection '{COLLECTION_NAME}' existante supprimée.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "dataset_hash": dataset_hash,
        },
    )

    # Vérification espace cosinus
    assert collection.metadata["hnsw:space"] == "cosine", (
        f"Espace attendu 'cosine', trouvé '{collection.metadata['hnsw:space']}'"
    )

    # Indexation par batch (ChromaDB limite à ~5461 par add)
    BATCH_SIZE = 1000
    for start in range(0, len(doc_ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(doc_ids))
        collection.add(
            ids=doc_ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings_list[start:end],
            metadatas=[
                {"title": titles[i], "token_count": token_counts[i]}
                for i in range(start, end)
            ],
        )
        print(f"  Batch {start}-{end} indexé.")

    # Vérification finale
    count = collection.count()
    assert count == EXPECTED_COUNT, f"Attendu {EXPECTED_COUNT} docs, trouvé {count}"

    print(f"\nIngestion terminée :")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Documents  : {count}")
    print(f"  Espace     : {collection.metadata['hnsw:space']}")
    print(f"  Hash       : {dataset_hash}")


if __name__ == "__main__":
    ingest()
