import os
import json
import argparse
import torch
import numpy as np
import faiss
import datasets
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
os.environ["HTTP_PROXY"] = "http://hacienda:3128"
os.environ["HTTPS_PROXY"] = "http://hacienda:3128" 

TOPK_BM25 = 100  # Number of initial documents from BM25
TOPK_FINAL = 10   # Number of final documents after reranking


def bm25_sphere_retrieval(data):
    from pyserini.search import LuceneSearcher

    index_path = os.environ.get("BM25_SPHERE_PATH")
    print("loading bm25 index, this may take a while...")
    searcher = LuceneSearcher(index_path)

    print("running bm25 retrieval...")
    for d in tqdm(data):
        query = d["question"]
        try:
            hits = searcher.search(query, TOPK)
        except Exception as e:
            # https://github.com/castorini/pyserini/blob/1bc0bc11da919c20b4738fccc020eee1704369eb/scripts/kilt/anserini_retriever.py#L100
            if "maxClauseCount" in str(e):
                query = " ".join(query.split())[:950]
                hits = searcher.search(query, TOPK)
            else:
                raise e

        docs = []
        for hit in hits:
            h = json.loads(str(hit.docid).strip())
            docs.append(
                {
                    "title": h["title"],
                    "text": hit.raw,
                    "url": h["url"],
                }
            )
        d["docs"] = docs

def load_wiki_docs():
    """Loads Wikipedia documents from DPR_WIKI_TSV file."""
    DPR_WIKI_TSV = os.environ.get("DPR_WIKI_TSV")
    print("Loading Wikipedia corpus...")
    docs = []
    with open(DPR_WIKI_TSV) as f:
        for i, line in enumerate(f):
            if i == 0:  # Skip header
                continue
            row = line.strip().split("\t")
            docs.append(row[2] + "\n" + row[1])  # title + text
    return docs


def bm25_retrieve(searcher, questions):
    """Retrieves top-100 documents using BM25."""
    retrieved_docs = []
    for query in tqdm(questions, desc="BM25 retrieval"):
        hits = searcher.search(query, TOPK_BM25)
        retrieved_docs.append([(hit.docid, hit.score) for hit in hits])
    return retrieved_docs


def gtr_rerank(encoder, queries, docs, bm25_results):
    """Encodes queries and reranks retrieved documents using GTR embeddings."""
    print("Encoding queries with GTR...")
    with torch.inference_mode():
        query_embs = encoder.encode(queries, batch_size=4, normalize_embeddings=True)

    print("Encoding BM25 retrieved docs with GTR...")
    doc_texts = [[docs[docid] for docid, _ in doc_list] for doc_list in bm25_results]
    doc_embs = [
        encoder.encode(doc_batch, batch_size=4, normalize_embeddings=True)
        for doc_batch in tqdm(doc_texts, desc="Encoding docs")
    ]

    # Convert to tensors for cosine similarity computation
    query_embs = np.array(query_embs, dtype=np.float32)
    doc_embs = [np.array(batch, dtype=np.float32) for batch in doc_embs]

    # Rank documents based on cosine similarity
    final_results = []
    for qi, q_emb in enumerate(query_embs):
        scores = np.dot(doc_embs[qi], q_emb)  # Compute cosine similarity
        topk_indices = np.argsort(scores)[::-1][:TOPK_FINAL]  # Get top-10

        final_docs = []
        for idx in topk_indices:
            doc_to_save = eval(doc_texts[qi][idx])
            doc_to_save["score"] = float(scores[idx])
            final_docs.append(doc_to_save)
        final_results.append(final_docs)
    return final_results


def hybrid_retrieval(data, dataset_name):
    """Hybrid BM25 + GTR retrieval for either MIRACL or Wikipedia."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer("sentence-transformers/gtr-t5-xxl", device=device)

    questions = [d["question"] for d in data]

    if dataset_name.lower() == "miracl":
        print("Using Pyserini BM25 for MIRACL...")
        searcher = LuceneSearcher.from_prebuilt_index("miracl-v1.0-en")
    elif dataset_name.lower() == "alce":
        print("Using Pyserini BM25 for DPR WIKI...")
        searcher = LuceneSearcher.from_prebuilt_index("wikipedia-dpr")
    else:
        raise ValueError("Invalid dataset. Choose 'miracl' or 'alce'.")
    bm25_results = bm25_retrieve(searcher, questions)
    print("Loading MIRACL corpus...")
    docs = {docid: searcher.doc(docid).raw() for hit_list in bm25_results for docid, _ in hit_list}
    #elif dataset_name.lower() == "alce":
        # print("Using BM25 for Wikipedia (DPR_WIKI_TSV)...")
        # docs = load_wiki_docs()
        # searcher = None  # No Pyserini index for Wiki, using in-memory BM25
        # bm25_results = bm25_sphere_retrieval(data)


    # Rerank with GTR
    reranked_results = gtr_rerank(encoder, questions, docs, bm25_results)

    # Update dataset with retrieved documents
    for qi, ret in enumerate(reranked_results):
        data[qi]["docs"] = ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid BM25 + GTR retrieval")
    parser.add_argument("--retriever", type=str, required=True, help="options: bm25_gtr")
    parser.add_argument("--dataset", type=str, default="alce", help="options: miracl/alce")
    parser.add_argument("--data_file", type=str, help="Path to the data file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--column_name", type=str, default="retrieved_docs", help="Column name to store  retrieved docs")
    args = parser.parse_args()

    extend_dataset = False
    if args.dataset.lower() == "miracl":
        data_name = "miracl/hagrid"
        dataset= datasets.load_dataset(
            data_name,
            split="dev",
        )
        
        data = [{"question": ex["query"]} for ex in dataset]
        extend_dataset = True
    else:
        with open(args.data_file) as f:
            data = json.load(f)

    if args.retriever == "bm25_gtr":
        hybrid_retrieval(data, args.dataset)
    else:
        raise NotImplementedError("Only 'bm25_gtr' retriever is implemented.")
    
    if extend_dataset: # Extend input doc with retrieved docs
        new_dataset = []
        for i,row in enumerate(dataset):
            assert(row["query"] == data[i]["question"])
            row[args.column_name] = data[i]["docs"]
            new_dataset.append(row)
        data = new_dataset
    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=4)
