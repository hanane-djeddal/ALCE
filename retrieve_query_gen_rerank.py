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
import transformers
from searcher import SearcherWithinDocs
import os

os.environ["HTTP_PROXY"] = "http://hacienda:3128"
os.environ["HTTPS_PROXY"] = "http://hacienda:3128"


def batch(docs: list, nb: int = 10):
    batches = []
    batch = []
    for d in docs:
        batch.append(d)
        if len(batch) == nb:
            batches.append(batch)
            batch = []
    if len(batch) > 0:
        batches.append(batch)
    return batches


def greedy_decode(model, input_ids, length, attention_mask, return_last_logits=True):
    decode_ids = torch.full(
        (input_ids.size(0), 1), model.config.decoder_start_token_id, dtype=torch.long
    ).to(input_ids.device)
    encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
    next_token_logits = None
    for _ in range(length):
        model_inputs = model.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            past=None,
            attention_mask=attention_mask,
            use_cache=True,
        )
        outputs = model(**model_inputs)  # (batch_size, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
        decode_ids = torch.cat(
            [decode_ids, next_token_logits.max(1)[1].unsqueeze(-1)], dim=-1
        )
    if return_last_logits:
        return decode_ids, next_token_logits
    return decode_ids


class MonoT5:
    def __init__(self, model_path="castorini/monot5-base-msmarco", device=None):
        self.model = self.get_model(model_path, device=device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "t5-base",
        )
        self.token_false_id = self.tokenizer.get_vocab()["▁false"]
        self.token_true_id = self.tokenizer.get_vocab()["▁true"]
        self.device = next(self.model.parameters(), None).device

    @staticmethod
    def get_model(
        pretrained_model_name_or_path: str, *args, device: str = None, **kwargs
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        return (
            transformers.AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
            .to(device)
            .eval()
        )

    def rerank(self, query, docs):
        d = self.rescore(query, docs)
        id_ = np.argsort([i["score"] for i in d])[::-1]
        return np.array(d)[id_]

    def rescore(self, query, docs):
        for b in batch(docs, 10):
            with torch.no_grad():
                text = [f'Query: {query} Document: {d["text"]} Relevant:' for d in b]
                model_inputs = self.tokenizer(
                    text,
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = model_inputs["input_ids"].to(self.device)
                attn_mask = model_inputs["attention_mask"].to(self.device)
                _, batch_scores = greedy_decode(
                    self.model,
                    input_ids=input_ids,
                    length=1,
                    attention_mask=attn_mask,
                    return_last_logits=True,
                )
                batch_scores = batch_scores[
                    :, [self.token_false_id, self.token_true_id]
                ]
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                batch_log_probs = batch_scores[:, 1].tolist()
            for doc, score in zip(b, batch_log_probs):
                doc["score"] = score  # dont update, only used as Initial with query
        return docs


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
    ranker = MonoT5(device="cuda")
    new_data = []
    for idx, item in enumerate(tqdm(data["data"])):
        # doc_list = item["docs"]
        if args.external_docs is not None:
            assert external[idx]["question"] == item["question"]
            doc_list = external[idx]["docs"]
        searcher = SearcherWithinDocs(
            doc_list, args.retriever, model=gtr_model, device=args.retriever_device
        )
        initial_docs = []
        for subquery in item["generated_text"]:
            print("Subquery : ", subquery)
            best_doc_id, scores = searcher.search_retrun_scores(subquery)
            best_doc_id = [int(d) for d in best_doc_id]
            scores = [float(s) for s in scores]
            print("Best doc:", best_doc_id[0])
            for docindice in best_doc_id:
                initial_docs.append(doc_list[docindice])

        ranked_doc = ranker.rerank(item["question"], initial_docs)[:10]
        item_result = {"reranked": True}
        for k in external[idx].keys():
            if k != "docs":
                item_result[k] = external[idx][k]
        item_result["docs"] = list(ranked_doc)
        new_data.append(item_result)
    tag = f".{args.retriever}"
    result_file_name = args.f.replace(
        ".csv", "_retrieved_docs" + tag + "_reranked.json"
    )
    json.dump(new_data, open(result_file_name, "w"), indent=4)


if __name__ == "__main__":
    main()
