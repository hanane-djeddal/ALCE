import os
os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'

from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_MODEL = "Qwen/Qwen3-8B" #"Qwen/Qwen3-4B"
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL)