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
from transformers import BitsAndBytesConfig
#from rouge_score import rouge_scorer, scoring
from tqdm import tqdm
import pandas as pd
import sys
import logging
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import math


from collections import defaultdict
import time

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from utils import normalize_answer, get_max_memory, remove_citations

# os.environ["HTTP_PROXY"] = "http://hacienda:3128"
# os.environ["HTTPS_PROXY"] = "http://hacienda:3128"
ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.append(ROOT_PATH)

from RAGnRoll.tools.eval_tools import read_json_files_from_folder, compute_metrics,concatenate_csv_files



os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'

AUTOAIS_MODEL = "google/t5_xxl_true_nli_mixture"
QWEN_MODEL = "Qwen/Qwen3-4B" #"Qwen/Qwen2.5-14B-Instruct-1M"

MAX_MODEL_LEN = 4096 
MAX_NEW_TOKENS = 1024

MAX_PROMPT_FOR_TRUNCATION = MAX_MODEL_LEN - MAX_NEW_TOKENS

global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None

global qwen_model, qwen_tokenizer
qwen_model, qwen_tokenizer = None, None



    # results_list = []

    # # 1. Split the data into smaller chunks (batches)
    # data_chunks = [data[i:i + BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]

    # # 2. Process each chunk (batch) sequentially
    # for batch in data_chunks:
    #     batch_results = compute_autoais(
    #         batch, 
    #         at_most_citations=args.at_most_citations,
    #         model_path=args.model_path
    #     )
    #     results_list.append(batch_results)
    # final_results = merge_results(results_list) 

    # # Update your final result dictionary
    # result.update(final_results)

    # result = {}
    # result["length"] = compute_len(normalized_data)
    # result.update(
    #     compute_autoais(
    #         data, at_most_citations=args.at_most_citations, model_path=args.model_path
    #     )
    # )


    # data_chunks = [
    #     data[i:i + BATCH_SIZE] 
    #     for i in range(0, len(data), BATCH_SIZE)
    # ]

    # # 4. Initialize results list
    # all_batch_results = []
    # total_batches = len(data_chunks)

    # # 5. Process batches one by one with cache clearing
    # print(f"Starting batch processing ({total_batches} batches, BATCH_SIZE={BATCH_SIZE})...")
    # for i, batch in enumerate(data_chunks):
    #     start_time = time.time()
        
    #     # --- A. RUN INFERENCE ---
    #     try:
    #         batch_result = compute_autoais(
    #             batch, 
    #             at_most_citations=args.at_most_citations
    #         )
    #         all_batch_results.append(batch_result)
    #     except RuntimeError as e:
    #         if "CUDA out of memory" in str(e):
    #             print("\n!!! CRITICAL CUDA OOM ERROR !!!")
    #             print(f"Batch {i+1} failed with BATCH_SIZE={BATCH_SIZE}.")
    #             print("You must reduce sequence length, use 4-bit quantization, or switch GPUs.")
    #             raise e
    #         else:
    #             raise e
                
    #     # --- B. CLEAR CACHE ---
    #     # This is the essential step you requested!
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
            
    #     end_time = time.time()
    #     print(f"Batch {i+1}/{total_batches} complete. Time taken: {end_time - start_time:.2f}s. CUDA cache cleared.")

    # # 6. Aggregate results (assuming results are dictionaries containing lists)
    # #    This logic flattens the results from all batches into a single dictionary.
    # if all_batch_results:
    #     final_autoais_result = defaultdict(list)
    #     for batch_dict in all_batch_results:
    #         for key, value in batch_dict.items():
    #             final_autoais_result[key].extend(value)
        
    #     # Update your main result dictionary (assuming 'result' is defined outside this block)
    #     # result.update(final_autoais_result) 
    #     print("All batches processed and results aggregated.")

def prepare_input(query,response,claim, passage,thinking=True):
    """
    """
    system_prompt = """You will be given a CLAIM and a DOCUMENT. Determine whether the claim is 'GROUNDED' or 'NOT GROUNDED' based on the document. A 'GROUNDED' claim is fully supported by the information provided in the document. It should be directly verifiable from the document. Only return the classification as the answer: 1 for 'GROUNDED' or 0 for 'NOT GROUNDED' without any explanation"""

    user_prompt = f"""CLAIM: {claim} 

    DOCUMENT: {passage}

    CLASSIFICATION:"""

    # system_prompt = """Your task is to determine whether a claim is 'GROUNDED' or 'NOT GROUNDED' based on a document. You will be given the "CLAIM" and the "DOCUMENT". The "CLAIM" is a part of a full answer in reponse to a question, you will also be given the full "ANSWER" as well as the "QUESTION" for full context. Use the "QUESTION" and the full "ANSWER" to fully contexualize the "CLAIM" then examine the "CLAIM" and the "DOCUMENT" to determine whether the "CLAIM" is grounded or not.  A 'GROUNDED' claim is fully supported by the information provided in the "DOCUMENT". It should be directly verifiable from the "DOCUMENT". Only return the classification as the your answer: 1 for 'GROUNDED' or 0 for 'NOT GROUNDED' without any explanation"""
    # user_prompt = f"""QUESTION:{query}

    # FULL ANSWER: {response}

    # CLAIM: {claim} 

    # DOCUMENT: {passage}

    # CLASSIFICATION:"""


    messages = [
            {
            "role": "system",
            "content": system_prompt,
            },
            {"role": "user", 
            "content": user_prompt}
        ]

    global qwen_model, qwen_tokenizer
    if qwen_model is None:
        qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
        qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    # Apply chat template
    text_input = qwen_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",tokenize=False,enable_thinking=thinking 
    )
    return text_input

def get_classification_confidence(raw_output, qwen_tokenizer):
    """Extract confidence for classification tokens"""
    logprobs_data = raw_output.outputs[0].logprobs
    token_ids = raw_output.outputs[0].token_ids
    
    for i, (token_id, logprob_dict) in enumerate(zip(token_ids, logprobs_data)):
        token = qwen_tokenizer.decode([token_id]).strip()
        if token in ['0', '1'] or 'GROUNDED' in token.upper():
            if logprob_dict and token_id in logprob_dict:
                return logprob_dict[token_id].logprob
    return None

def generate_vllm(all_inputs,scored=None):
    global qwen_model, qwen_tokenizer

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS,
        truncate_prompt_tokens=MAX_PROMPT_FOR_TRUNCATION,
        logprobs=scored,
    )
    all_outputs=qwen_model.generate(all_inputs, sampling_params) # [inputs]

    all_answers=[]
    all_logprobs=[]
    for idx_output, raw_output in enumerate(all_outputs):
        output = raw_output.outputs[0].text
        
        try:
            # rindex finding 151668 (</think>)
            index_txt = len(output) - output.index("</think>")
        except ValueError:
            index_txt = 0
        try:
            # rindex finding 151668 (</think>)
            index = len(raw_output.outputs[0].token_ids) - raw_output.outputs[0].token_ids[::-1].index(151668)
        except ValueError:
            index = 0


        # thinking_content = output[:index_txt].strip("\n")
        # content = output[index_txt:].strip("\n")

        # print("txtttt thinking content:", thinking_content)
        # print("content:", content)


        thinking_content = qwen_tokenizer.decode(raw_output.outputs[0].token_ids[:index], skip_special_tokens=True).strip("\n")
        content = qwen_tokenizer.decode(raw_output.outputs[0].token_ids[index:], skip_special_tokens=True).strip("\n")
        classification_score = 0.0 #-float('inf')
        if scored:
            logprobs_data = raw_output.outputs[0].logprobs
            #print("logprobs_data",logprobs_data)
            
        
            for i in range(index, len(logprobs_data)):
                token_id = list(logprobs_data[i].keys())[0]
                if index < len(logprobs_data):
                    token_logprob_dict = logprobs_data[i]
                    
                    if token_id in token_logprob_dict:
                        token_obj = token_logprob_dict[token_id]
                        decoded_token = token_obj.decoded_token
                        
                        # CHECK: Is this token just whitespace?
                        if decoded_token and decoded_token.strip():
                            log_prob = token_obj.logprob
                            classification_score = math.exp(log_prob)
                            print("token found",decoded_token,classification_score)
                            # if '1' not in decoded_token.upper() or 'GROUNDED' not in decoded_token.upper():
                            #     print("!!! Different than class")

                            break 
                # first_answer_token_logprobs = logprobs_data[index:]
                
                # # Get the specific token ID that was actually generated
                # generated_token_id = raw_output.outputs[0].token_ids[index:][-2]                
            print("logprobs:", classification_score)
        print("thinking content:", thinking_content)
        print("content:", content)



        
        # Decode and extract result
        result = content #tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        result = result.strip()
        if '1' in result or 'GROUNDED' in result.upper():
            filetred_answer= 1
        else:
            filetred_answer= 0
        all_answers.append(filetred_answer)
        all_logprobs.append(classification_score)
        
        # if logprobs_data:
        #     print(f"Sample {idx_output} logprobs:")
        #     for token_id, logprob_dict in zip(raw_output.outputs[0].token_ids, logprobs_data):
        #         token = qwen_tokenizer.decode([token_id])
        #         # logprob_dict is a dict mapping token_id to Logprob object
        #         if logprob_dict:
        #             for tid, lp in logprob_dict.items():
        #                 print(f"  Token: '{token}' (id={tid}), logprob: {lp.logprob:.4f}")
        
        # print("Using get_classification_confidence:",get_classification_confidence(raw_output,qwen_tokenizer))
    return all_answers, all_logprobs


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
    input_ids = autoais_tokenizer(input_text, return_tensors="pt",truncation=True, max_length=512).input_ids.to(
        autoais_model.device
    )
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference


def _run_qwen(passage, claim):
    """
    """
    if passage == '' or claim== '':
        return 0
    #system_prompt = """You will be given a claim and a document. Determine whether the claim is 'GROUNDED' or 'NOT GROUNDED' based on the document.A 'GROUNDED' claim is factually accurate and fully supported by the information provided in the document. It should be directly verifiable from the document. Only return the classification as the answer: 1 for 'GROUNDED' or 0 for 'NOT GROUNDED' without any explanation"""
    system_prompt = """Your task is to determine whether a claim is 'GROUNDED' or 'NOT GROUNDED' on a document. You will be given the "CLAIM" and the "DOCUMENT". The "CLAIM" is a part of a full answer in reponse to a question, you will also be given the full "ANSWER" as well as the "QUESTION" for full context. Use the "QUESTION" and the full "ANSWER" to fully contexualize the "CLAIM" then examine the "CLAIM" and the "DOCUMENT" to determine whether the "CLAIM" is grounded or not.  A 'GROUNDED' claim is fully supported by the information provided in the "DOCUMENT". It should be directly verifiable from the "DOCUMENT". Only return the classification as the your answer: 1 for 'GROUNDED' or 0 for 'NOT GROUNDED' without any explanation"""

    user_prompt = f"""CLAIM: {claim} 

    DOCUMENT: {passage}

    CLASSIFICATION:"""

    messages = [
            {
            "role": "system",
            "content": system_prompt,
            },
            {"role": "user", 
            "content": user_prompt}
        ]

    global qwen_model, qwen_tokenizer
    if qwen_model is None:
        qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
        qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    # Apply chat template
    text_input = qwen_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    inputs = qwen_tokenizer(text_input, return_tensors="pt").to(qwen_model.device)
 
    with torch.inference_mode():
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=1024,  # We only need 1 token (0 or 1)
        )
        output_ids  = generated_ids[0][len(inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = qwen_tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = qwen_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)
    
    # Decode and extract result
    result = content #tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    result = result.strip()
    if '1' in result or 'GROUNDED' in result.upper():
        return 1
    else:
        return 0

def compute_entail_llm(
    data,
    decontext=False,
    concat=False,
    at_most_citations=None,
    model_path=None,
    filter_column=None,
    filter_value=None,
    scored=False,
    vllm=False,
    batch=100,
    thinking=True,
    prediction_column='auto_score',
):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    global qwen_model, qwen_tokenizer
    if qwen_model is None:
        if model_path is None:
            model_name = QWEN_MODEL
            logger.info("Loading QWEN model...")
        else:
            logger.info(f"Loading custom model...{model_path}")
            model_name = model_path
        
        if vllm:
            qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)

            qwen_model = LLM(
                model=model_name,
                tensor_parallel_size=torch.cuda.device_count(),
                gpu_memory_utilization=0.9,
                max_model_len=MAX_MODEL_LEN,
                dtype='bfloat16',
                enforce_eager=False,
                # for LORA
                trust_remote_code=True,
                #enable_lora=True,
            )
        else:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16, #float32, #bfloat16, #
                dtype=torch.bfloat16, #float32, #bfloat16, 
            )


            qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
            qwen_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16, #torch_dtype
                #max_memory=get_max_memory(),
                device_map="auto",
                quantization_config=nf4_config
            )

    logger.info(f"Running QWEN...")
    get_logits= 1 if scored else None
    logger.info(f"Getting LOGITS {scored} top {get_logits}...")

    ais_scores = []
    ais_scores_prec = []


    #autoais_log = []
    updated_items=[]
    accuracy=[]

    prepared_inputs=[]
    batched_items=[]
    for idx,item in tqdm(enumerate(data)):
        if filter_column is not None and filter_value is not None:
            if filter_column in item.keys():
                if item[filter_column]!=filter_value:
                    updated_items.append(item)
                    continue
        # Get sentences by using NLTK
        sents = item["claim"]
        if len(sents) == 0:
            continue
        target_sent = remove_citations(sents).strip()

        
        if vllm:
            query = (
                item["question"]
                if 'question' in item.keys() and item["question"] not in ["nan", "", None]
                else ""
            )
            response = (
                item["response"]
                if "response" in item.keys() and item["response"] not in ["nan", "", None]
                else ""
            )
            claim = (
                item["claim"]
                if item["claim"] and item["claim"] not in ["nan", "", None]
                else ""
            )
            claim = remove_citations(claim).strip()
            refs = [ str(ref)  if ref not in ["nan", "", None] else "" for ref in item["references"]]
            documents_concatenation = "\n\n\n".join(refs)

            prepared_inputs.append(prepare_input(query,response,claim, documents_concatenation, thinking))
            batched_items.append(item)
            if idx != 0 and idx%batch == 0:
                res, probs = generate_vllm(prepared_inputs, scored=get_logits)
                for idx_res in range(len(res)):
                    ais_scores.append(res[idx_res]) 
                    batched_items[idx_res][prediction_column] = res[idx_res]
                    if scored:
                        batched_items[idx_res]["logit"] = probs[idx_res]
                    int_gold_label= 1 if (batched_items[idx_res]['attribution_label']== "attributable") else 0
                    batched_items[idx_res]["accuracy"] = 1 if batched_items[idx_res][prediction_column] == int_gold_label else 0
                    accuracy.append(batched_items[idx_res]["accuracy"])
                    updated_items.append(batched_items[idx_res])

                prepared_inputs=[]
                batched_items=[]
        else:     
            if scored:
                nli_result = _run_qwen("\n".join(item["references"]), target_sent)
                item["logit"] = 0
            else:
                nli_result = _run_qwen("\n".join(item["references"]), target_sent)
            # autoais_log.append(
            # {
            #     "question": item["question"], ##question query
            #     "claim": item["claim"],
            #     "passage": item["references"],
            #     "model_type": "NLI",
            #     "model_output": nli_result,
            #     }
            # )

            ais_scores.append(nli_result) 
            item[prediction_column] = nli_result
            int_gold_label= 1 if (item['attribution_label']== "attributable") else 0
            item["accuracy"] = 1 if item[prediction_column] == int_gold_label else 0
            accuracy.append(item["accuracy"])
            updated_items.append(item)
    return {
        "ais_scores": 100 * np.mean(ais_scores),
        "accuracy": 100 *np.mean(accuracy),
        "data":updated_items,
    }


# def _run_qwen(passage, claim):
#     """
#     Run inference for assessing AIS between a premise and hypothesis.
#     Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
#     """
#     global qwen_model, qwen_tokenizer
#     if qwen_model is None:
#          #"stabilityai/stablelm-zephyr-3b" #"meta-llama/Llama-2-7b-chat-hf"
#         logger.info(f"Loading Language model...{QWEN_MODEL}")
#         # Load the model and tokenizer
#         qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL) #, cache_dir= os.environ['WORK'] + '/.cache/huggingface/hub')
#         qwen_model = LLM(
#             model=QWEN_MODEL,
#             tensor_parallel_size=torch.cuda.device_count(),
#             gpu_memory_utilization=0.9,
#             max_model_len=8192,
#             dtype='bfloat16',
#             enforce_eager=False,
#             # for LORA
#             trust_remote_code=True,
#             enable_lora=True,
#         )
#     input_text = [
#             {
#                 "role": "system",
#                 "content": system_prompt,
#             },
#             {"role": "user", 
#             "content": user_prompt}
#         ]
#     inputs = qwen_tokenizer.apply_chat_template(
#             input_text, add_generation_prompt=True, return_tensors="pt",tokenize=False
#         )
#     input_text = "premise: {} hypothesis: {}".format(passage, claim)
#     input_ids = autoais_tokenizer(input_text, return_tensors="pt",truncation=True, max_length=512).input_ids.to(
#         autoais_model.device
#     )
#     with torch.inference_mode():
#         outputs = autoais_model.generate(input_ids, max_new_tokens=10)
#     result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     inference = 1 if result == "1" else 0
#     return inference
    
def _run_nli_autoais_scored(passage, claim):
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(passage, claim)    
    try:
        # For some tokenizers, "0" and "1" might be represented as single tokens.
        # It's crucial to check if your tokenizer tokenizes them as expected.
        token_id_0 = autoais_tokenizer.encode("0", add_special_tokens=False)[0]
        token_id_1 = autoais_tokenizer.encode("1", add_special_tokens=False)[0]
    except IndexError:
        print("Warning: '0' or '1' might not be single tokens in your tokenizer. Adjust logic if needed.")
        # Fallback for demonstration if "0" or "1" are not single tokens, though this is less robust.
        # In a real scenario, you'd need to ensure your model outputs single tokens for labels.
        token_id_0 = None
        token_id_1 = None

    input_ids = autoais_tokenizer(input_text, return_tensors="pt",truncation=True, max_length=512).input_ids.to(
        autoais_model.device
    )

    with torch.inference_mode():
        outputs = autoais_model.generate(
            input_ids,
            max_new_tokens=1,  # We only expect '0' or '1' as output, so only generate 1 token
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,   # Use greedy decoding for a deterministic output and its probability
            num_beams=1        # Ensure greedy search
        )

    # The generated sequence will contain the predicted token (0 or 1)
    # outputs.sequences is typically (batch_size, input_length + num_generated_tokens)
    generated_token_id = outputs.sequences[0, -1].item() # Get the last (and only) generated token ID

    # The scores will contain the logits for the first (and only) generated token
    # outputs.scores is a tuple, where each element corresponds to a generated token.
    # Since max_new_tokens=1, outputs.scores will have one element.
    logits_for_first_token = outputs.scores[0] # Shape: (batch_size, vocab_size)

    # Assuming batch_size = 1, get the logits for the specific token IDs "0" and "1"
    prob_0 = 0.0 # Default if token_id_0 not found
    prob_1 = 0.0 # Default if token_id_1 not found

    if token_id_0 is not None and token_id_1 is not None:
        # Apply softmax to convert logits to probabilities
        probabilities = torch.softmax(logits_for_first_token, dim=-1)

        # Get the probability for token "0" and token "1"
        prob_0 = probabilities[0, token_id_0].item()
        prob_1 = probabilities[0, token_id_1].item()
    else:
        # Fallback if "0" or "1" are not single tokens.
        # In this case, we can only get the probability of the *generated* token.
        print("Could not directly calculate prob for '0' and '1'. Calculating prob for generated token.")
        probabilities = torch.softmax(logits_for_first_token, dim=-1)
        if generated_token_id == token_id_0:
            prob_0 = probabilities[0, generated_token_id].item()
        elif generated_token_id == token_id_1:
            prob_1 = probabilities[0, generated_token_id].item()


    # Determine the predicted label
    result_token = autoais_tokenizer.decode(generated_token_id, skip_special_tokens=True)
    inference = 1 if result_token == "1" else 0 # Assuming "1" for positive, "0" for negative

    # Determine the probability associated with the *predicted* inference label
    predicted_label_probability = 0.0
    if inference == 1:
        predicted_label_probability = prob_1
    else:
        predicted_label_probability = prob_0

    return inference, predicted_label_probability

# def compute_claims(data):
#     global autoais_model, autoais_tokenizer
#     if autoais_model is None:
#         logger.info("Loading AutoAIS model...")
#         autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
#             AUTOAIS_MODEL,
#             torch_dtype=torch.bfloat16,
#             max_memory=get_max_memory(),
#             device_map="auto",
#         )
#         autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

#     logger.info("Computing claims...")
#     scores = []
#     for item in tqdm(data):
#         normalized_output = remove_citations(item["output"])
#         entail = 0
#         claims = item["claims"]
#         for claim in claims:
#             entail += _run_nli_autoais(normalized_output, claim)
#         scores.append(entail / len(claims))
#     return 100 * np.mean(scores)


def compute_autoais(
    data,
    decontext=False,
    concat=False,
    at_most_citations=None,
    model_path=None,
    filter_column=None,
    filter_value=None,
    scored=False,
    prediction_column='auto_score',
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
        if model_path is None:
            model_name = AUTOAIS_MODEL
            logger.info("Loading AutoAIS model...")
        else:
            logger.info(f"Loading custom model...{model_path}")
            model_name = model_path
        

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, #float32, #bfloat16, #
            dtype=torch.bfloat16, #float32, #bfloat16, 
        )


        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16, #torch_dtype
            #max_memory=get_max_memory(),
            device_map="auto",
            quantization_config=nf4_config
        )
        autoais_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    logger.info(f"Running AutoAIS...")


    ais_scores = []
    ais_scores_prec = []


    #autoais_log = []
    updated_items=[]
    accuracy=[]
    for item in tqdm(data):
        if filter_column is not None and filter_value is not None:
            if filter_column in item.keys():
                if item[filter_column]!=filter_value:
                    updated_items.append(item)
                    continue
        # Get sentences by using NLTK
        sents = item["claim"]
        if len(sents) == 0:
            continue

        target_sent = remove_citations(sents).strip()
        if scored:
            nli_result, predicted_label_probability = _run_nli_autoais_scored("\n".join(item["references"]), target_sent)
            item["logit"] = predicted_label_probability
        else:
            nli_result = _run_nli_autoais("\n".join(item["references"]), target_sent)
        # autoais_log.append(
        # {
        #     "question": item["question"], ##question query
        #     "claim": item["claim"],
        #     "passage": item["references"],
        #     "model_type": "NLI",
        #     "model_output": nli_result,
        #     }
        # )

        ais_scores.append(nli_result) 
        item[prediction_column] = nli_result
        int_gold_label= 1 if (item['attribution_label']== "attributable") else 0
        item["accuracy"] = 1 if item[prediction_column] == int_gold_label else 0
        accuracy.append(item["accuracy"])
        updated_items.append(item)
    return {
        "ais_scores": 100 * np.mean(ais_scores),
        "accuracy": 100 *np.mean(accuracy),
        "data":updated_items,
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
        "--split", type=str, help="dataset split", choices=["train", "dev", "test","test_ood"], default=None
    )

    parser.add_argument(
        "--dataset", type=str, help="dataset", choices=["hagrid", "verifiability","attriBench","true"], default=None
    )

    parser.add_argument(
        "--dataset_name", type=str, help="dataset", choices=["full_data", "subset_balanced"], default=None
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
        "--eval_model", type=str, help="which eval model", choices=["nli_true_zs", "alignscore","nli_aligned","custom","llm"], default="nli_true_zs"
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to the metric model", default=None
    )
    parser.add_argument(
        "--scored", action="store_true", help="Get proba logits"
    )
    parser.add_argument(
        "--code_validation", action="store_true", help="Run code validation checks"
    )
    parser.add_argument(
        "--results_folder", type=str, default="results/metric_eval/", help="Results folder"
    )
    parser.add_argument(
        "--true_folder", type=str, default="/lustre/fswork/projects/rech/fiz/udo61qq/Code/true/", help="True benchamrk folder"
    )
    parser.add_argument(
        "--tag", type=str, default=None, help="tag"
    )
    parser.add_argument(
        "--resume_from_file", type=str, default=None, help="Resume from file"
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=None,
        help="Batch size",
    )
    parser.add_argument(
        "--batchid",
        type=int,
        default=None,
        help="number of specific batch",
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
    parser.add_argument(
        "--filter_column", type=str, default=None, help="Only eval if filter_column satisfy cnd"
    )
    parser.add_argument(
        "--filter_value", type=str, default=None, help="Cnd for filtering"
    )
    parser.add_argument(
        "--concat_files", action="store_true", help="concatenate files"
    )
    parser.add_argument(
        "--without_file_update", action="store_true", help="update file"
    ) 
    parser.add_argument(
        "--evaluate", action="store_true",  help="Evaluate results",
    )
    parser.add_argument(
        "--group_by_column",type=str, default="src_dataset", help=" group_by_column"
    )
    
    parser.add_argument(
        "--prediction_column",type=str, default="auto_score", help="predictin column"
    )
    parser.add_argument(
        "--vllm", action="store_true",  help="use vllm",
    )
    parser.add_argument(
        "--without_thinking", action="store_false",  help="don't use thinking mode",
    )
    parser.add_argument(
        "--merge", action="store_true",  help="merge files",
    )


    args = parser.parse_args()
    #################
    # Logging params
    #################
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)} - {parser.get_default(arg)}")

    code_validation="code_validation_" if args.code_validation else ""
    tag= code_validation+args.tag +"_" if args.tag else code_validation
    tag=tag+args.dataset +"_" if args.dataset else tag
    tag=tag+args.dataset_name +"_" if args.dataset_name else tag
    tag=tag+args.eval_model+"_" if args.eval_model else tag
    tag=tag+"trained_" if args.model_path else tag
    tag=tag+'withoutthinking_' if args.without_thinking==False else tag
    tag=tag+"scored_" if args.scored else tag
    tag=tag+args.split if args.split else tag
    results_file=code_validation+"metric_eval_" +tag 
    results_folder=os.path.join(args.results_folder , tag )
    if args.f is not None:
        results_folder_name=os.path.dirname(args.f)
        results_folder=os.path.join(results_folder_name , tag )

    print("Evaluating file:", args.f)

    try:
        logger.info(f"MAKING NEW FOLDER: {args.results_folder}")
        # Use os.makedirs() to create all necessary parent directories 
        # and the final directory itself.
        # exist_ok=True prevents an error if the directory already exists.
        os.makedirs(results_folder, exist_ok=True)
        logger.info(f"Directory '{results_folder}' created successfully or already exists.")
    except Exception as e:
        logger.info(f"An error occurred: {e}")


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
            if "params" in data_with_config.keys():
                params= data_with_config["params"]

    if args.dataset == "hagrid":
        data = datasets.load_dataset("miracl/hagrid", split=args.split)
    elif args.dataset == "verifiability" and args.f is not None:
        with open(args.f, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    elif args.dataset == "true" and args.true_folder is not None:
        data = concatenate_csv_files(args.true_folder)
        # Display the head of the final dataset
        column_mapping = {
            'label': 'attribution_label',
            'generated_text': 'claim',
            'grounding':'references'
            # Add all other columns you want to rename here
        }
        data.rename(columns=column_mapping, inplace=True)
        data['references']=data["references"].apply(lambda x: [x])
        data['attribution_label']=data["attribution_label"].apply(lambda x: "attributable" if x ==1 else "not attributable")
        if data is not None:
            print(data.head())
            print(data['references'][0])
            data=data.to_dict('records')
        print("True Data sample:",data[0])
        print("True Data sample-claim:",data[0]["claim"])
        print("True Data length:",len(data))
        

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
        if args.dataset_name == "subset_balanced":
            data_path=os.environ['WORK']+ "/AttributionBench"
        elif args.dataset_name =="full_data":
            data_path=os.environ['WORK']+ "/AttributionBenchfull_data"

        data = datasets.load_from_disk(data_path)
        data = data[args.split] #, split=args.split, features=features)
        print(data)
        data = data.to_list()


        # data = datasets.load_dataset("osunlp/AttributionBench", name=args.dataset_name, split=args.split, features=features)
        # print(data)
        # Truncate by newline and remove on the fly search result
    logger.warning(
        "We remove all the pre/appended space/newlines and we truncate the answer by the first newline."
    )
    logger.warning(
        "We replace any on the fly search result to standard bracket citation format."
    )
    for i in range(len(data)):
        data[i]["claim"] = str(data[i]["claim"]).strip().split("\n")[0]
        data[i]["claim"] = data[i]["claim"].replace("<|im_end|>", "")
        if "query" in data[i].keys():
            data[i]["question"] = data[i]["query"]


 
    # Remove all citations for all non-AutoAIS evaluation
    normalized_data = copy.deepcopy(data)
    for i in range(len(normalized_data)):
        normalized_data[i]["claim"] = remove_citations(normalized_data[i]["claim"])

    # Prepared BATCHED data
    if args.batchsize is not None and args.batchid is not None:
        BATCH_SIZE = args.batchsize 
        start = args.batchid*BATCH_SIZE
        end=(args.batchid+1)*BATCH_SIZE
        logger.info(f"Batch ID {args.batchid} sized {BATCH_SIZE}")
        logger.info(f"Itertaing from {start} to {end}")
        full_dataset_length=len(data)
        data=data[start:end]
        normalized_data=normalized_data[start:end]


    # Run METRIC
    result = {}
    result["length"] = compute_len(normalized_data)
    if args.eval_model == "llm":
        result.update(
            compute_entail_llm(
                data, at_most_citations=args.at_most_citations, model_path=args.model_path, filter_column=args.filter_column, filter_value=args.filter_value, scored=args.scored, vllm=args.vllm,thinking=args.without_thinking, prediction_column=args.prediction_column
            )
        )
    else:
        result.update(
            compute_autoais(
                data, at_most_citations=args.at_most_citations, model_path=args.model_path, filter_column=args.filter_column, filter_value=args.filter_value, scored=args.scored, prediction_column=args.prediction_column
            )
        )


    # Accuracy Eval
    if args.evaluate:
        print("Evaluation Results of:", args.eval_model, args.model_path)
        #df = pd.DataFrame(result["data"])
        try:
            all_scores=compute_metrics(result["data"],prediction_column=args.prediction_column, scoredlabels=args.scored,group_by_column=args.group_by_column)

            #result["auc_roc"]=all_scores[2].to_dict('index')
            result["evaluation2"]=all_scores[0].to_dict('index') 
            result["evaluation3"]=all_scores[1].to_dict('index')
        except:
            logger.info("Error while Evaluating")


    merged_result= None
    merged= False
    if args.batchsize is not None and args.batchid is not None:
        merged_result= None
        if end >= full_dataset_length and args.merge:
            # if args.f:
            #     folder_path = os.path.dirname(args.f)
            # else:
            folder_path=results_folder
            logger.info(f"Merging All Batched Files in folder {folder_path}")
            try:
                merged_result = read_json_files_from_folder(folder_path)
                result["data"] = merged_result["data"] + result["data"]
                logger.info(f"All merged results: {len(result['data'])}")
                merged=True
            except:
                logger.info("Error while Merging Files")
                merged_result=None
            if args.evaluate and merged_result:
                print("Evaluation Results of:", args.eval_model, args.model_path)
                #df = pd.DataFrame(result["data"])
                try:
                    all_scores=compute_metrics(result["data"],prediction_column=args.prediction_column, scoredlabels=args.scored,group_by_column=args.group_by_column)

                    #result["auc_roc"]=all_scores[2].to_dict('index')
                    result["evaluation2"]=all_scores[0].to_dict('index') 
                    result["evaluation3"]=all_scores[1].to_dict('index')
                    #result["data"]= merged_result
                except:
                    logger.info("Error while Evaluating")

    if args.dataset == "true" and args.true_folder is not None:
        if merged_result:
            df = pd.DataFrame(merged_result["data"]) 
            df["score"]=df.apply(lambda x : 1-x["logit"] if x[args.prediction_column]==0 else x["logit"],axis=1)
            df["label"]=df.apply(lambda x : 1 if x['attribution_label']=="attributable" else 0,axis=1)
            for src in df["src_dataset"].unique():
                subdf=df[df["src_dataset"] == src]
                subresults_file=results_file[:-5]+src+".csv"
                save_csv=os.path.join(args.true_folder, subresults_file)
                subdf.to_csv(save_csv)
                print("Saving results True df:",save_csv)
                
            save_csv=os.path.join(args.true_folder, results_file)
            df.to_csv(save_csv, index=False)
            print("Saving results True df:",save_csv)

    # if args.f:
    #     if args.batchsize is not None and args.batchid is not None:
    #         results_file=args.f[:-5]+str(start)+"-"+str(end)+"nli.json"
    #     else:
    #         results_file=args.f[:-5]+"_nli.json"

    if args.batchsize is not None and args.batchid is not None and not merged:
        results_file= str(start)+"-"+str(end) + results_file+".json"
    else:
        results_file= results_file+".json"
    results_file = os.path.join(results_folder, results_file)

    logger.info(f"Saving Result to {results_file}")
    with open(results_file, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
