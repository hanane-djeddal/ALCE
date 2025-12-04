#!/bin/sh
# Script JEANZAY

############ GENERAL SETTINGS ############
#SBATCH --job-name=true_eval
#SBATCH --array=0-4 # 
#SBATCH --output=logs/nodrw_%A_%a.out
#SBATCH --error=logs/nodrw_%A_%a.err
##########################################




#SBATCH --gres=gpu:4   #for v account max is 4 #
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=2:00:00
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=30
#SBATCH --account=fiz@a100
#SBATCH -C a100

################### V100 default ######################
##SBATCH -C v100
##SBATCH -C v100-16g                 # decommenter pour reserver uniquement des GPU V100 16 Go
##SBATCH -C v100-32g                 # decommenter pour reserver uniquement des GPU V100 32 Go
##SBATCH -A cfh@v100
##SBATCH --cpus-per-task=3           # nombre de CPU par tache pour gpu_p2 (1/8 du noeud 8-GPU)
##SBATCH --partition=gpu_p2
##SBATCH --qos=qos_gpu-t3
###################       ######################


################### V100 32Go ######################
##SBATCH -C v100-32g                 # decommenter pour reserver uniquement des GPU V100 32 Go
##SBATCH --cpus-per-task=3           # nombre de CPU par tache pour gpu_p2 (1/8 du noeud 8-GPU)
##SBATCH --partition=gpu_p2
##SBATCH --qos=qos_gpu-t3
###################       ######################


################### A100 80Go ######################
##SBATCH -C a100                     # decommenter pour la partition gpu_p5 (GPU A100 80 Go)
##SBATCH --cpus-per-task=8           # nombre de CPU par tache pour gpu_p5 (1/8 des CPU du noeud 8-GPU A100)
##SBATCH --partition=gpu_p5
##SBATCH --qos=qos_gpu_a100-t3 
##SBATCH --qos=qos_gpu_a100-dev	

###################       ######################

################### H100 80Go ######################
##SBATCH -C h100                     # decommenter pour la partition gpu_p5 (GPU A100 80 Go)
##SBATCH -A cfh@h100
##SBATCH --cpus-per-task=24           # nombre de CPU par tache pour gpu_p6 (1/4 des CPU du noeud 4-GPU H100)
##SBATCH --partition=gpu_p6
##SBATCH --qos=qos_gpu_h100-t3
###################       ######################


######################################################



export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module purge
module load arch/a100
module load pytorch-gpu/py3/2.8.0
export HF_HOME=$WORK/.cache/huggingface


#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench"  --split "test" --batchsize 400 --batchid  $SLURM_ARRAY_TASK_ID 
#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --f /lustre/fswork/projects/rech/fiz/udo61qq/Code/RAGnRoll/results/augmented_datasets/Llama-3.1-8B-Instruct_allfewshot/all_augmented_datasetLlama-3.1-8B-Instruct_allfewshot_corrected.json --filter_column "example_type" --filter_value generated --tag "generated_augmented" --batchsize 15000 --batchid  $SLURM_ARRAY_TASK_ID 

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "true" --true_folder "/lustre/fswork/projects/rech/fiz/udo61qq/Code/true/" --scored --evaluate --batchsize 15000 --batchid  $SLURM_ARRAY_TASK_ID 

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "true" --true_folder "/lustre/fswork/projects/rech/fiz/udo61qq/Code/true/" --evaluate --batchsize 15000 --batchid  $SLURM_ARRAY_TASK_ID --eval_model "llm" --tag  "scored_qwen3-4B_ft_unsloth_qwen_classif_3ep_true_16bit" --vllm  --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/2ep_attribench_ogargs_3ep_16bit --scored

CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "true" --true_folder "/lustre/fswork/projects/rech/fiz/udo61qq/Code/true/" --evaluate --batchsize 15000 --batchid  $SLURM_ARRAY_TASK_ID --eval_model "llm" --tag  "scored_qwen3-4B_zs" --vllm  --scored


#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --f /lustre/fswork/projects/rech/fiz/udo61qq/Code/RAGnRoll/results/augmented_datasets/Llama-3.1-8B-Instruct_allfewshot_v01/all_augmented_datasetLlama-3.1-8B-Instruct_allfewshot_v01.json  --filter_column "example_type" --filter_value generated --group_by_column "error_type" --tag "train_gen_evaluated" --eval_model "nli_aligned" --scored --evaluate --batchsize 15000 --batchid  $SLURM_ARRAY_TASK_ID 