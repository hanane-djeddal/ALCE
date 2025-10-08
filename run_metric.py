import argparse
import collections
import json
import re
import string
import torch
import copy
import pandas as pd
import os
from nltk import sent_tokenize
import numpy as np
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm
import pandas as pd
import sys
import logging
import datasets

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from utils import normalize_answer, get_max_memory, remove_citations

os.environ["HTTP_PROXY"] = "http://hacienda:3128"
os.environ["HTTPS_PROXY"] = "http://hacienda:3128"


AUTOAIS_MODEL = "google/t5_xxl_true_nli_mixture"

global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None

def compute_len(data):
    """Compute average length of predictions."""

    res, cntr = 0, 0
    for item in data:
        res += len(item["claim"].split())
        cntr += 1
    return res / cntr

def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(
        autoais_model.device
    )
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference


def compute_claims(data):
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
            AUTOAIS_MODEL,
            torch_dtype=torch.bfloat16,
            max_memory=get_max_memory(),
            device_map="auto",
        )
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info("Computing claims...")
    scores = []
    for item in tqdm(data):
        normalized_output = remove_citations(item["output"])
        entail = 0
        claims = item["claims"]
        for claim in claims:
            entail += _run_nli_autoais(normalized_output, claim)
        scores.append(entail / len(claims))
    return 100 * np.mean(scores)


def compute_autoais(
    data,
    decontext=False,
    concat=False,
    at_most_citations=None,
):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
            AUTOAIS_MODEL,
            torch_dtype=torch.bfloat16,
            max_memory=get_max_memory(),
            device_map="auto",
        )
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info(f"Running AutoAIS...")


    ais_scores = []
    ais_scores_prec = []


    autoais_log = []
    updated_items=[]
    for item in tqdm(data):
        # Get sentences by using NLTK
        sents = item["claim"]
        if len(sents) == 0:
            continue

        target_sents = remove_citations(sent).strip()

        nli_result = _run_nli_autoais("\n".join(item["references"]), target_sent)
        autoais_log.append(
        {
            "question": item["question"], ##question query
            "claim": item["claim"],
            "passage": item["references"],
            "model_type": "NLI",
            "model_output": nli_result,
            }
        )

        ais_scores.append(nli_result) 
        item["autoais_score"] = nli_result
        updated_items.append(item)
    return {
        "ais_scores": 100 * np.mean(ais_scores),
        "all_ais_scores": ais_scores,
        "all_items":updated_items,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--f",
        type=str,
        default=None,
        help="Output file. Should have field `question`, `output`, (ROUGE) `answer`, \
                        (accuracy) `qa_pairs`, (AIS) `docs`",
    )

    parser.add_argument(
        "--split", type=str, help="dataset split", choices=["train", "dev", "test","test_ood"], default="train"
    )

    parser.add_argument(
        "--dataset", type=str, help="dataset", choices=["hagrid", "verifiability","attriBench"], default="hagrid"
    )

    parser.add_argument(
        "--dataset_name", type=str, help="dataset", choices=["full_data", "subset_balanced"], default="subset_balanced"
    )
    
    parser.add_argument(
        "--citations", action="store_true", help="Evaluation with citation"
    )
    parser.add_argument(
        "--at_most_citations",
        type=int,
        default=3,
        help="At most take this many documents (mostly for precision)",
    )

    parser.add_argument(
        "--eval_model", type=str, help="which eval model", choices=["nli_true_zs", "alignscore","nli_true_aligned","custom","nli_true_zs_scored","nli_true_aligned_scored"], default="nli_true_zs"
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to the metric model", default=None
    )
    parser.add_argument(
        "--code_validation", action="store_true", help="Run code validation checks"
    )
    parser.add_argument(
        "--results_folder", type=str, default="results/metric_eval/", help="Results folder"
    )
    parser.add_argument(
        "--tag", type=str, default=None, help="tag"
    )
    parser.add_argument(
        "--resume_from_file", type=str, default=None, help="Resume from file"
    )
    parser.add_argument(
        "--startindex",
        type=int,
        default=0,
        help="Index to start iterations",
    )
    parser.add_argument(
        "--stopindex",
        type=int,
        default=None,
        help="Index to stop iterations",
    )

    args = parser.parse_args()
    #################
    # Logging params
    #################
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)} - {parser.get_default(arg)}")

    code_validation="code_validation_" if args.code_validation else ""
    tag= args.tag if args.tag else ""
    tag= tag + "_" +args.dataset_name if args.dataset_name else tag
    results_file=code_validation+"metric_eval_"+args.dataset + "_" + args.eval_model + "_" +tag + "_" + args.split
    results_folder=os.path.join(args.results_folder , args.dataset + "_" + args.eval_model + "_" +tag + "_" + args.split)
    print("Evaluating file:", args.f)

    if args.f is not None:
        if args.f.endswith(".csv"):
            df = pd.read_csv(
                    args.f,
                    converters={
                        "qa_pairs": eval,
                        "wikipages": eval,
                    "annotations": eval,
                    "docs": eval,
                },
            )
            data = df.to_dict("records")
        else:
            with open(args.f) as f:
                data_with_config = json.load(f)
            data = data_with_config["data"]

    if args.dataset == "hagrid":
        data = datasets.load_dataset("miracl/hagrid", split=args.split)
    elif args.dataset == "verifiability" and args.f is not None:
        with open(args.f, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    elif args.dataset == "attriBench":
        features = datasets.Features({
        'question': datasets.Value('string'),
        'claim': datasets.Value('string'),
        'claim_raw_string': datasets.Value('string'),
        'response': datasets.Value('string'),
        'references': datasets.Sequence(datasets.Value("string")),
        'citation_links': datasets.Sequence(datasets.Value("string")),
        'webpage_references': datasets.Sequence(datasets.Value("string")),
        'attribution_label': datasets.Value('string'),
        'src_dataset': datasets.Value('string'),
        'id': datasets.Value('string'),
        })

        data = datasets.load_dataset("osunlp/AttributionBench", name=args.dataset_name, split=args.split, features=features)
        print(data)
        # Truncate by newline and remove on the fly search result
        logger.warning(
            "We remove all the pre/appended space/newlines and we truncate the answer by the first newline."
        )
        logger.warning(
            "We replace any on the fly search result to standard bracket citation format."
        )
        for i in range(len(data)):
            data[i]["claim"] = data[i]["claim"].strip().split("\n")[0]
            data[i]["claim"] = data[i]["claim"].replace("<|im_end|>", "")
            if "query" in data[i].keys():
                data[i]["question"] = data[i]["query"]




        # Remove all citations for all non-AutoAIS evaluation
        normalized_data = copy.deepcopy(data)
        for i in range(len(normalized_data)):
            normalized_data[i]["claim"] = remove_citations(normalized_data[i]["claim"])

    result = {}
    result["length"] = compute_len(normalized_data)
    result.update(
        compute_autoais(
            data, at_most_citations=args.at_most_citations
        )
    )
    # if args.claims_nli:
    #     result["claims_nli"] = compute_claims(normalized_data)


    print(result)
    results_file = os.path.join(results_folder, results_file)
    with open(results_file, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
