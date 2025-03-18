#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=hagridalce ##alceprompt # le nom du job (voir commande squeue)
#SBATCH --nodes=1 # le nombre de noeuds 
#SBATCH --nodelist=zz
#SBATCH --gpus=1 # nombre de gpu
#SBATCH --ntasks-per-node=1 # nombre de tache par noeud 
#SBATCH --time=1-90:00:00             # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=jz_%j_%x.out     # nom du fichier de sortie
#SBATCH --error=errjz_%j_%x.out      # nom du fichier d'erreur (ici commun avec la sortie)

# Source l'environement par example ~/.bashrc
source ~/.bashrc
# activer l'environement python
conda activate llms-env #selfrag
cd /home/djeddal/Documents/Code/ALCE



#python eval.py --citations --qa --mauve --f  /home/djeddal/Documents/Code/results_jz/13b_seg_proba_att_only/v5_with_intruct/all_testasqa_llama-2-chat-hagrid-att-seg-proba-rag-agent-13b_instruction_prompt_using_answer_for_retrieval_8rounds_3docs.json
python eval.py --citations --qa --mauve --f  /home/djeddal/Documents/Code/results_jz/13b_seg_proba_att_only/v4_forced_ret_withanswer/all_testHagrid_forced_rounds_retrival_with_answer_seg_13b_llama_corrected_v100_using_answer_for_retrieval__8rounds_3docs.json


#python retrieval_bm25gtr_hagrid_alce.py --retriever bm25_gtr --dataset miracl --data_file queries.json --output_file hagrid_retrieve_bm25_gtr.json

#python run.py --config configs/asqa_llama-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 5 --shot 0 --model HuggingFaceH4/zephyr-7b-beta --tag reranked_query_gen --eval_file results_query_gen/generated_queries_4shot_4q_asqa_llama_retrieved_docs.gtr-t5-large_reranked.json
#python run.py --config configs/asqa_alpaca-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 0 --shot 0 --model meta-llama/Llama-2-13b-chat-hf --prompt_file prompts/asqa_closedbook.json
#python run.py --config configs/asqa_llama-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 5 --shot 0 --model meta-llama/Llama-2-13b-chat-hf
#python post_hoc_cite.py --f result/asqa-Llama-2-13b-chat-hf-gtr_light_inst-shot0-ndoc0-42.json --external_docs data/asqa_eval_gtr_top100.json


#python run.py --config configs/asqa_llama-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 5 --shot 0 --model HuggingFaceH4/zephyr-7b-beta --tag reranked_query_gen --eval_file results_query_gen/generated_queries_4shot_4q_asqa_llama_retrieved_docs.gtr-t5-large_reranked.json
#python run.py --config configs/asqa_alpaca-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 0 --shot 0 --model HuggingFaceH4/zephyr-7b-beta --prompt_file prompts/asqa_closedbook.json
#python run.py --config configs/asqa_llama-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 5 --shot 0 --model HuggingFaceH4/zephyr-7b-beta

#python post_hoc_cite.py --f result/asqa-zephyr-7b-beta-gtr_light_inst-shot0-ndoc0-42.json --external_docs data/asqa_eval_gtr_top100.json
#python eval.py --citations --qa --mauve --f /home/djeddal/Documents/Code/Attributed-IR/results/RTG_vanilla/generation_RTG_vanilla_2_passages_corrected.json
#python run.py --config configs/asqa_llama-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 5 --shot 0 --model meta-llama/Llama-2-7b-chat-hf --tag reranked_query_gen --eval_file results_query_gen/generated_queries_4shot_4q_asqa_llama_retrieved_docs.gtr-t5-large_reranked.json
#python run.py --config configs/asqa_alpaca-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 0 --shot 0 --model stabilityai/stablelm-zephyr-3b --prompt_file prompts/asqa_closedbook.json