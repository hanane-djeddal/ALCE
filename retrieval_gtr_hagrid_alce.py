import os
import json
import argparse
import torch
import faiss
import numpy as np
import csv
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

os.environ["HTTP_PROXY"] = "http://hacienda:3128"
os.environ["HTTPS_PROXY"] = "http://hacienda:3128"
TOPK = 10  # Number of retrieved documents


def gtr_build_index(encoder, docs, index_path):
    """Encodes documents and stores them in a FAISS index."""
    with torch.inference_mode():
        embs = encoder.encode(
            docs, batch_size=4, show_progress_bar=True, normalize_embeddings=True
        ).astype("float32")  # FAISS requires float32

    # Build FAISS index
    dimension = embs.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
    index.add(embs)

    # Save the FAISS index
    faiss.write_index(index, index_path)
    return index


def load_miracl_docs():
    """Loads documents from the MIRACL dataset."""
    print("Loading MIRACL dataset...")
    miracl = load_dataset("miracl/miracl", "en")
    return [doc["text"] for doc in miracl["train"]]  # Use 'text' field


def load_alce_docs():
    """Loads documents from the DPR Wikipedia TSV file."""
    print("Loading ALCE (DPR Wikipedia) dataset...")
    DPR_WIKI_TSV = os.environ.get("DPR_WIKI_TSV")
    docs = []
    with open(DPR_WIKI_TSV) as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            docs.append(row[2] + "\n" + row[1])  # Combine title & text
    return docs


def gtr_retrieval(data, dataset_name):
    """Performs retrieval using the selected dataset (MIRACL or ALCE)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading GTR encoder...")
    encoder = SentenceTransformer("sentence-transformers/gtr-t5-xxl", device=device)

    questions = [d["question"] for d in data]
    with torch.inference_mode():
        queries = encoder.encode(
            questions, batch_size=4, show_progress_bar=True, normalize_embeddings=True
        ).astype("float32")

    # Load dataset based on user selection
    if dataset_name == "miracl":
        docs = load_miracl_docs()
        index_path = "gtr_miracl_index.faiss"
    elif dataset_name == "alce":
        docs = load_alce_docs()
        index_path = "gtr_alce_index.faiss"
    else:
        raise ValueError("Invalid dataset. Choose 'miracl' or 'alce'.")

    # Load or build FAISS index
    if not os.path.exists(index_path):
        print(f"FAISS index not found for {dataset_name}, building...")
        index = gtr_build_index(encoder, docs, index_path)
    else:
        print(f"FAISS index found for {dataset_name}, loading...")
        index = faiss.read_index(index_path)

    del encoder  # Free GPU memory

    print(f"Running GTR retrieval on {dataset_name}...")
    queries = np.array(queries, dtype="float32")
    _, indices = index.search(queries, TOPK)  # FAISS top-k search

    # Retrieve top-k documents
    for qi, idx_list in enumerate(indices):
        ret = []
        for i in idx_list:
            ret.append({"id": str(i + 1), "text": docs[i]})
        data[qi]["docs"] = ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Passage retrieval.")
    parser.add_argument("--retriever", type=str, required=True, help="options: bm25/gtr")
    parser.add_argument("--dataset", type=str,  default="alce", help="options: miracl/alce")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file with retrieved docs.")
    args = parser.parse_args()

    with open(args.data_file) as f:
        data = json.load(f)

    if args.retriever == "gtr":
        gtr_retrieval(data, args.dataset.lower())
    else:
        raise NotImplementedError("Only 'gtr' retriever is implemented.")

    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=4)
