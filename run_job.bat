#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=eli5 ###trainedwoutq2 ##alceprompt # le nom du job (voir commande squeue)
#SBATCH --nodes=1 # le nombre de noeuds 
#SBATCH --nodelist=zz
#SBATCH --gpus=2 # nombre de gpu
#SBATCH --ntasks-per-node=1 # nombre de tache par noeud 
#SBATCH --time=48:00:00             # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=jz_%j_%x.out     # nom du fichier de sortie
#SBATCH --error=errjz_%j_%x.out      # nom du fichier d'erreur (ici commun avec la sortie)

# Source l'environement par example ~/.bashrc
source ~/.bashrc
# activer l'environement python
conda activate selfrag #llms-env #selfrag
cd /home/djeddal/Documents/Code/ALCE



#python eval.py --citations --qa --mauve --f  /home/djeddal/Documents/Code/results_jz/13b_seg_proba_att_only/v5_with_intruct/all_testasqa_llama-2-chat-hagrid-att-seg-proba-rag-agent-13b_instruction_prompt_using_answer_for_retrieval_8rounds_3docs.json
#python eval.py --citations --qa --mauve --f  /home/djeddal/Documents/Code/results_jz/13b_seg_proba_att_only/v4_forced_ret_withanswer/all_testHagrid_forced_rounds_retrival_with_answer_seg_13b_llama_corrected_v100_using_answer_for_retrieval__8rounds_3docs.json


#python retrieval_bm25gtr_hagrid_alce.py --retriever bm25_gtr --dataset miracl --data_file queries.json --output_file hagrid_retrieve_bm25_gtr.json

#python run.py --config configs/asqa_llama-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 5 --shot 0 --model HuggingFaceH4/zephyr-7b-beta --tag reranked_query_gen --eval_file results_query_gen/generated_queries_4shot_4q_asqa_llama_retrieved_docs.gtr-t5-large_reranked.json
#python run.py --config configs/asqa_alpaca-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 0 --shot 0 --model meta-llama/Llama-2-13b-chat-hf --prompt_file prompts/asqa_closedbook.json
#python run.py --config configs/asqa_llama-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 5 --shot 0 --model meta-llama/Llama-2-13b-chat-hf
#python post_hoc_cite.py --f result/asqa-Llama-2-13b-chat-hf-gtr_light_inst-shot0-ndoc0-42.json --external_docs data/asqa_eval_gtr_top100.json

### eli5
#python run.py --config configs/asqa_alpaca-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 0 --shot 0 --model meta-llama/Llama-2-13b-chat-hf --prompt_file prompts/asqa_closedbook.json --eval_file data/eli5_eval_bm25_top100_with_ids.json --dataset_name eli5
#python run.py --config configs/asqa_llama-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 5 --shot 0 --model meta-llama/Llama-2-13b-chat-hf --eval_file data/eli5_eval_bm25_top100_with_ids.json --dataset_name eli5
#python post_hoc_cite.py --f result/eli5-Llama-2-13b-chat-hf-gtr_light_inst-shot0-ndoc0-42.json --external_docs data/eli5_eval_bm25_top100_with_ids.json 


#python run.py --config configs/asqa_llama-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 5 --shot 0 --model HuggingFaceH4/zephyr-7b-beta --tag reranked_query_gen --eval_file results_query_gen/generated_queries_4shot_4q_asqa_llama_retrieved_docs.gtr-t5-large_reranked.json
#python run.py --config configs/asqa_alpaca-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 0 --shot 0 --model HuggingFaceH4/zephyr-7b-beta --prompt_file prompts/asqa_closedbook.json
#python run.py --config configs/asqa_llama-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 5 --shot 0 --model HuggingFaceH4/zephyr-7b-beta

#python post_hoc_cite.py --f result/asqa-zephyr-7b-beta-gtr_light_inst-shot0-ndoc0-42.json --external_docs data/asqa_eval_gtr_top100.json
#python eval.py --citations --qa --mauve --f /home/djeddal/Documents/Code/Attributed-IR/results/RTG_vanilla/generation_RTG_vanilla_2_passages_corrected.json
#python run.py --config configs/asqa_llama-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 5 --shot 0 --model meta-llama/Llama-2-7b-chat-hf --tag reranked_query_gen --eval_file results_query_gen/generated_queries_4shot_4q_asqa_llama_retrieved_docs.gtr-t5-large_reranked.json
#python run.py --config configs/asqa_alpaca-7b_shot2_ndoc3_gtr_light_inst.yaml --ndoc 0 --shot 0 --model stabilityai/stablelm-zephyr-3b --prompt_file prompts/asqa_closedbook.json

#python eval.py --citations  --f  /home/djeddal/Documents/Code/results_jz/13b_seg_proba_att_inf_NoQ/all_testHagrid_llama-2-chat-hagrid-att-seg-proba-rag-agent-13b_8rounds_3docsempty_query.json

#python eval.py --citations  --f  /home/djeddal/Documents/Code/results_jz/13b_param_att_only/v2_forcing_rounds/all_testasqa_llama-2-chat-hagrid-att-param-knw-rag-agent-13b_8rounds_3docs.json
#python eval.py --citations  --f  /home/djeddal/Documents/Code/results_jz/13b_stat_att_only/v2_forcing_rounds/all_testasqa_llama-2-chat-hagrid-att-rag-agent-13b_8rounds_3docs.json

#python eval.py --citations  --f  /home/djeddal/Documents/Code/results_jz/13b_param_att_only/v2_forcing_rounds/all_testHagrid_llama-2-chat-hagrid-att-param-knw-rag-agent-13b_8rounds_3docs.json
#python eval.py --citations  --f  /home/djeddal/Documents/Code/results_jz/13b_param_att_only/all_testasqa_llama-2-chat-hagrid-att-param-knw-rag-agent-13b_4rounds_3docs.json

#python eval.py --citations  --f /home/djeddal/Documents/Code/results_jz/13b_param_att_only/v2_forcing_rounds/all_testasqa_llama-2-chat-hagrid-att-param-knw-rag-agent-13b_8rounds_3docs.json
#python eval.py --citations  --f  /home/djeddal/Documents/Code/results_jz/13b_param_att_only/v2_forcing_rounds/all_testHagrid_llama-2-chat-hagrid-att-param-knw-rag-agent-13b_8rounds_3docs.json

#python eval.py --citations  --f  /home/djeddal/Documents/Code/results_jz/13b_stat_att_only/v2_forcing_rounds/all_testHagrid_llama-2-chat-hagrid-att-rag-agent-13b_8rounds_3docs.json

#python eval.py --citations  --f  /home/djeddal/Documents/Code/results_jz/13b_stat_att_only/v2_forcing_rounds/all_testasqa_llama-2-chat-hagrid-att-rag-agent-13b_8rounds_3docs.json

#python eval.py --citations  --f  /home/djeddal/Documents/Code/RAGnRoll/results/segmentation/adjusted_for_evalall_testHagrid_forced_rounds_retrival_with_answer_seg_13b_llama_corrected_v100_using_answer_for_retrieval__8rounds_3docsfaithful_eval_proba_sentence.json

### friday
#python eval.py --citations  --f  /home/djeddal/Documents/Code/results_jz/13b_seg_proba_att_only/v12_appending_user_query/all_testHagrid_llama-2-chat-hagrid-att-seg-proba-rag-agent-13b_8rounds_3docs.json
#python eval.py --citations  --f /home/djeddal/Documents/Code/results_jz/13b_seg_proba_att_only/v11_nli_seg_train/all_testHagrid_llama-2-chat-hagrid-att-segnli-rag-agent-13b_using_answer_for_retrieval__8rounds_3docs.json

#python eval.py --citations  --f /home/djeddal/Documents/Code/results_jz/13b_seg_proba_att_only/v11_nli_seg_train/all_testasqa_llama-2-chat-hagrid-att-segnli-rag-agent-13b_using_answer_for_retrieval_8rounds_3docs.json

#python eval.py --citations  --f /home/djeddal/Documents/Code/results_jz/13b_decom_wout_query_att/all_testHagrid__llama-2-chat-hagrid-att-without-query-rag-agent-13b_4rounds_3docswithout_query.json
#python eval.py --citations  --f /home/djeddal/Documents/Code/results_jz/13b_decom_wout_query_att/all_testasqa__4rounds_3docswithout_query.json


#python eval.py --citations  --f /home/djeddal/Documents/Code/RAGnRoll/results/segmentation/adjusted_all_testHagrid_llama-2-chat-hagrid-attributable-13b_4rounds_3docssftfaithful_eval_proba_removing_sent.json
#python eval.py --citations  --f /home/djeddal/Documents/Code/RAGnRoll/results/segmentation/adjustedselfrag_hagrid_with_retrieval_5docs_13b_gtrbm25_corrected_citationsfaithful_eval_mixed_sentence.json

#python eval.py --citations  --claims_nli --f  /home/djeddal/Documents/Code/results_jz/eli5/all_testasqa_eli5_seg_llama13_using_answer_for_retrieval_8rounds_3docs.json

#python eval.py --citations  --f /home/djeddal/Documents/Code/RAGnRoll/results/segmentation/adjustedselfrag_hagrid_with_retrieval_5docs_13b_gtrbm25_corrected_citationsfaithful_eval_proba_removing_sent.json

#python eval.py --citations  --f  /home/djeddal/Documents/Code/results_jz/13b_seg_proba_att_inf_NoQ/all_testHagrid_llama-2-chat-hagrid-att-seg-proba-rag-agent-13b_2rounds_3docsempty_query.json


#python  eval.py --citations  --f  /home/djeddal/Documents/Code/results_jz/13b_decom_wout_query_att/v1_gtr/all_testHagrid_llama-2-chat-hagrid-att-without-query-rag-agent-13b_4rounds_3docswithout_query.json

#python  eval.py --citations  --f /home/djeddal/Documents/Code/results_jz/13b_seg_proba_att_inf_NoQ/v1_gtr/all_testHagrid_llama-2-chat-hagrid-att-seg-proba-rag-agent-13b_8rounds_3docsempty_query.json

#python eval.py --citations  --f /home/djeddal/Documents/Code/results_jz/13b_seg_proba_att_only/v12_appending_user_query/all_testasqa_llama-2-chat-hagrid-att-seg-proba-rag-agent-13b_appendinguseruquery_8rounds_3docs.json
#python eval.py --citations  --f  /home/djeddal/Documents/Code/results_jz/13b_decom_wout_query_att/v1_gtr/all_testHagrid_llama-2-chat-hagrid-att-without-query-rag-agent-13b_2rounds_3docswithout_query.json

#python eval.py --citations  --f /home/djeddal/Documents/Code/RAGnRoll/results/segmentation/adjustedall_testHagrid_llama-2-chat-hagrid-att-seg-proba-rag-agent-13b_8rounds_3docsempty_queryfaithful_eval_proba_removing_sent.json

#python eval.py --citations  --f  /home/djeddal/Documents/Code/RAGnRoll/results/segmentation/adjustedgeneration_RTG_vanilla_2_passages_jsonfaithful_eval_proba_removing_sent.json

#python eval.py --citations  --f /home/djeddal/Documents/Code/results_jz/13b_decom_wout_query_att/v1_gtr/all_testHagrid_llama-2-chat-hagrid-att-without-query-rag-agent-13b_2rounds_3docswithout_query.json

#python eval.py --citations  --f /home/djeddal/Documents/Code/results_jz/13b_decom_wout_query_att/v1_gtr/deduplicatedcorrecting1ststatall_testHagrid_llama-2-chat-hagrid-att-without-query-rag-agent-13b_4rounds_3docswithout_query.json

#python eval.py --citations  --f /home/djeddal/Documents/Code/results_jz/13b_decom_wout_query_att/v1_gtr/deduplicatedall_testHagrid_llama-2-chat-hagrid-att-without-query-rag-agent-13b_4rounds_3docswithout_query.json

#python eval.py --citations  --f  /home/djeddal/Documents/Code/results_jz/13b_decom_wout_query_att/v1_gtr/correcting1ststatall_testHagrid_llama-2-chat-hagrid-att-without-query-rag-agent-13b_4rounds_3docswithout_query.json
#python eval.py --citations  --f  /home/djeddal/Documents/Code/results_jz/13b_seg_proba_att_only/v12_appending_user_query/all_testHagrid_llama-2-chat-hagrid-att-seg-proba-rag-agent-13b_2rounds_3docs.json


#python eval.py --citations  --f /home/djeddal/Documents/Code/results_jz/13b_seg_proba_att_only/v12_appending_user_query/all_testHagrid_llama-2-chat-hagrid-att-seg-proba-rag-agent-13b_oneround_4th_3docs.json

#python eval.py --citations  --f /home/djeddal/Documents/Code/self-rag/retrieval_lm/selfrag_13b_eli5_with_retrieval_eval_gtr_top100_5docs.json
#python eval.py --citations  --f /home/djeddal/Documents/Code/ALCE/result/eli5-Llama-2-13b-chat-hf-gtr_light_inst-shot0-ndoc0-42post_hoc_citegtr-t5-large-external.json ##/home/djeddal/Documents/Code/ALCE/result/eli5-Llama-2-13b-chat-hf-gtr_light_inst-shot0-ndoc5-42.json ##/home/djeddal/Documents/Code/self-rag/retrieval_lm/selfrag_hagrid_with_retrieval_3docs_13b_gtrbm25.json

#python eval.py --citations  --f /home/djeddal/Documents/Code/RAGnRoll/results/segmentation/all_testHagrid_llama-2-chat-hagrid-att-seg-proba-rag-agent-13b_8rounds_3docsempty_queryfaithful_eval_multi_round_eval_proba_removing_sent.json
#python eval.py --citations  --f /home/djeddal/Documents/Code/RAGnRoll/results/segmentation/all_testHagrid_llama-2-chat-hagrid-att-seg-proba-rag-agent-13b_8rounds_3docsempty_queryfaithful_eval_proba_removing_sent.json
#python eval.py --citations  --f /home/djeddal/Documents/Code/RAGnRoll/results/segmentation/all_testHagrid_llama-2-chat-hagrid-att-seg-proba-rag-agent-13b_8rounds_3docsfaithful_eval_multi_round_eval_proba_removing_sent.json
#python eval.py --citations  --f /home/djeddal/Documents/Code/RAGnRoll/results/segmentation/all_testHagrid_llama-2-chat-hagrid-att-seg-proba-rag-agent-13b_8rounds_3docsfaithful_eval_multi_round_eval_using_subquery_proba_proba_removing_sent.json


#python eval.py --citations  --f  /home/djeddal/Documents/Code/Search-in-the-Chain/searchchain_hagrid_results_adjuted2.json
#python eval.py --citations  --f  /home/djeddal/Documents/Code/Search-in-the-Chain/searchchain_alce_eli5_results_adjuted2.json
#python eval.py --citations  --f  /home/djeddal/Documents/Code/Search-in-the-Chain/searchchain_alce_results_adjuted2.json

#python eval.py --citations  --f /home/djeddal/Documents/Code/results_jz/13b_seg_proba_att_only/v13_retrieve_once/all_testHagrid_llama-2-chat-hagrid-att-seg-proba-rag-agent-13bGTR_retrieve_once_8rounds_3docs.json


python run_metric.py --dataset "attriBench"  --split "test"  --dataset_name "full_data"