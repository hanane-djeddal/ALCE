import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from nltk import sent_tokenize
import re
import numpy as np
import pandas as pd
import string
import torch
from searcher import SearcherWithinDocs
import os

os.environ["HTTP_PROXY"] = "http://hacienda:3128"
os.environ["HTTPS_PROXY"] = "http://hacienda:3128"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, help="Output data file")
    parser.add_argument(
        "--retriever",
        type=str,
        default="gtr-t5-large",
        help="Retriever to use. Options: `tfidf`, `gtr-t5-large`",
    )
    parser.add_argument(
        "--retriever_device",
        type=str,
        default="cuda",
        help="Where to put the dense retriever if using. Options: `cuda`, `cpu`",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing citations"
    )
    parser.add_argument(
        "--external_docs", type=str, default=None, help="Use external documents"
    )

    args = parser.parse_args()
    if args.f.endswith(".csv"):
        df = pd.read_csv(
            args.f,
            converters={
                "generated_text": eval,
            },
        )
        data = {"data": df.to_dict("records")}
    else:
        data = json.load(open(args.f))
    new_data = []
    if args.external_docs is not None:
        external = json.load(open(args.external_docs))

    # Load retrieval model
    if "gtr" in args.retriever:
        from sentence_transformers import SentenceTransformer

        gtr_model = SentenceTransformer(
            f"sentence-transformers/{args.retriever}", device=args.retriever_device
        )
    new_data = []
    for idx, item in enumerate(tqdm(data["data"])):
        # doc_list = item["docs"]
        if args.external_docs is not None:
            assert external[idx]["question"] == item["question"]
            doc_list = external[idx]["docs"]
        searcher = SearcherWithinDocs(
            doc_list, args.retriever, model=gtr_model, device=args.retriever_device
        )

        for subquery in item["generated_text"]:
            print("Subquery : ", subquery)
            best_doc_id, scores = searcher.search_retrun_scores(subquery)
            best_doc_id = [int(d) for d in best_doc_id]
            scores = [float(d) for d in scores]
            print("Best doc:", best_doc_id[0])
            new_data.append(
                {
                    "question": item["question"],
                    "sub_query": subquery,
                    "retrieved_passages": best_doc_id,
                    "scores": scores,
                }
            )
    tag = f".{args.retriever}"
    result_file_name = args.f.replace(".csv", "with_retrieved_docs" + tag + ".json")
    json.dump(new_data, open(result_file_name, "w"), indent=4)


if __name__ == "__main__":
    main()
