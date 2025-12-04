#!/bin/sh
# Script JEANZAY

#SBATCH --job-name=deva100train_4Bvllmprompt2 #ft #qwen3-8B #vllm_eval #metric_ckpt #gendata #sftmixed #truebach2 #attribench_trained_balanced_testoodchpt300#trained_balanced_testood ata_gen_eval #
#SBATCH --output=jz_%j_%x.out    # nom du fichier de sortie
#SBATCH --error=errjz_%j_%x.out  
#SBATCH --gres=gpu:4  #for v account max is 4 #
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=1:50:00
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=30
#SBATCH --account=fiz@a100
#SBATCH -C a100

export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module purge
module load arch/a100
module load pytorch-gpu/py3/2.8.0
export HF_HOME=$WORK/.cache/huggingface

cd /lustre/fswork/projects/rech/fiz/udo61qq/Code/ALCE/


#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench"  --split "test" --eval_model "nli_aligned" --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_aligned_augmented2kv05ep
#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench"  --split "test_ood"  --eval_model "nli_aligned" --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_aligned_augmented2kv05ep

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench"  --split "test_ood"  --eval_model "nli_aligned" --tag "veryfirst" --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_aligned

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench"  --split "test_ood"  --eval_model "nli_aligned" --tag "2kgendata" --model_path  /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_aligned_augmented2kv0
#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "full_data" --split "test" --eval_model "nli_aligned" --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_aligned_augmented2kv05ep
#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "full_data" --split "test_ood"  --eval_model "nli_aligned" --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_aligned_augmented2kv05ep

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag "2kgendata30ckpt" --model_path  /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models//lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_aligned_augmented2kv0/checkpoint-300
#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test"  --eval_model "nli_aligned" --tag  "allgendataposonly2e300ckpt" --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_aligned_augmentedhardposgenbalanced2ep/checkpoint-300

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "allgendataposonly2e300ckpt" --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_aligned_augmentedhardposgenbalanced2ep/checkpoint-300

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "true" --true_folder "/lustre/fswork/projects/rech/fiz/udo61qq/Code/true/" --scored --evaluate --batchsize 15000 --batchid  1

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test"  --eval_model "nli_aligned" --tag  "sftmixeddatachpt4500" --scored --evaluate  --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_sft_augmentedmixedtraingen/checkpoint-4500

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "sftmixeddatachpt4500" --scored --evaluate  --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_sft_augmentedmixedtraingen/checkpoint-4500

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "sftmixeddata" --scored --evaluate  --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_sft_augmentedmixedtraingen


#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --f "/lustre/fswork/projects/rech/fiz/udo61qq/Code/RAGnRoll/results/augmented_datasets/Llama-3.1-8B-Instruct_allfewshot_v01/all_augmented_datasetLlama-3.1-8B-Instruct_allfewshot_v01_dedubfiltered.json" --filter_column "example_type" --filter_value generated --group_by_column "error_type"  --tag "evaluated" --eval_model "nli_aligned" --scored --evaluate

### oct 30th

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test"  --eval_model "nli_aligned" --tag  "classifaugmentedv1posandgenonlywithdub"  --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_classif_augmentedv2_posandgenonly/checkpoint-600 --scored --evaluate

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test"  --eval_model "nli_aligned" --tag  "classifogset600ckpt"  --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_attributionBench_classiftuned_OGsetv2/checkpoint-600 --scored --evaluate

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test"  --eval_model "nli_aligned" --tag  "classifogset100ckpt"  --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_attributionBench_classiftuned_originalset/checkpoint-100 --scored --evaluate

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "classifaugmentedv1_shuffled_mixedalltrain100ckpt"  --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_classif_augmentedv1_shuffled_mixedalltrain/checkpoint-100 --scored --evaluate

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "classifaugmentedv1_posandgenonly_dedub200ckpt"  --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_classif_augmentedv2_posandgenonly_dedub/checkpoint-200  --scored --evaluate

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "classifaugmentedv0SelectErrormixedtraingen300ckpt"  --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_classifsft_augmentedSelectErrormixedtraingen/checkpoint-300  --scored --evaluate


###nov 5th

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "classifOGsetv2"  --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_attributionBench_classiftuned_OGsetv2 --scored --evaluate

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "classifaugmentedv0errorselect2k5" --scored --evaluate --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_classifsft_augmentedSelectErrormixedtraingen/checkpoint-2500

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "classifaugmentedv1posandgen1k5" --scored --evaluate --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_classif_augmentedv2_posandgenonly_dedub/checkpoint-1500


###nov 6th
#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "classifaugmentedshuffled_mixedallfiltered2900" --scored --evaluate --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_classif_augmentedv1_shuffled_mixedalltrainfiltereddev2/checkpoint-900

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "classifaugmentedshuffled_mixedallfilteredfull2k4" --scored --evaluate --model_path  /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_classif_augmentedv1_shuffled_mixedalltrainfilteredfull/checkpoint-2400


#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "classifaugmentedshuffled_posandgenonly_dedubfull3k" --scored --evaluate --model_path  /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_classif_augmentedv2_posandgenonly_dedubfull/checkpoint-3400

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "classifaugmentedshuffled_mixedallfiltered2ch1k6" --scored --evaluate --model_path  /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_classif_augmentedv1_shuffled_mixedalltrainfiltereddev2/checkpoint-1600


### nov 7th : sft vs classif training code 
#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "testingsfttrainfiltered2ch100" --scored --evaluate --model_path  /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_sft_augmentedv1_shuffled_mixedalltrainfiltered2
#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "testingsfttrainOGsetch200" --scored --evaluate --model_path  /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_attributionBench_sft_OGset/checkpoint-200


#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "llm" --tag  "qwen3-4B_prompt2" --evaluate --model_path "Qwen/Qwen3-4B" --vllm

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "train"  --eval_model "llm" --tag  "qwen3-4B" --evaluate --model_path "Qwen/Qwen3-4B"

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "train"  --eval_model "llm" --tag  "qwen_3_4Bprompt2" --evaluate --vllm  #--without_thinking --model_path "Qwen/Qwen3-8B"


CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "llm" --tag  "qwen3-4B_ft_unsloth_qwen_classiftrl_2ep_attribench_16bit" --evaluate --vllm  --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/2ep_attribenchaugmented_ogargs_2ep_devsplit_16bit
#/lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/qwen3_4b_attribench_classif_unsloth_2e_trl_ogargs_16bit
#/lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/2ep_attribench_ogargs_3ep_16bit
#/lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/2ep_attribenchaugmented_ogargs_2ep_devsplit_16bit
#/lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/2ep_attribench_ogargs_3ep_16bit
#/lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/unsloth_qwen_classif_2ep_attribench_16bit
#/lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/2ep_attribench_ogargs_16bit
#/lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/2ep_attribench_ogargs_16bit
#/lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/qwen3_4b_attribench_classif_unsloth_10ep_trl_augmentedv1_shuffled_mixedalltrainfiltered/checkpoint-1038
#"/lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/qwen_classif_trained_16bit_2ep"
#/lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/qwen_classif_trained_16bit
#"/lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/qwen3_4b_attribench_sft"


/lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/2ep_attribench_ogargs_3ep_16bit