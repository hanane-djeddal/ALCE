#!/bin/sh
# Script JEANZAY

#SBATCH --job-name=a00_metric_sftmixed #truebach2 #attribench_trained_balanced_testoodchpt300#trained_balanced_testood
#SBATCH --output=jz_%j_%x.out    # nom du fichier de sortie
#SBATCH --error=errjz_%j_%x.out  
#SBATCH --gres=gpu:4  #for v account max is 4 
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=2:00:00
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

CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "true" --true_folder "/lustre/fswork/projects/rech/fiz/udo61qq/Code/true/" --scored --evaluate --batchsize 15000 --batchid  1

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test"  --eval_model "nli_aligned" --tag  "sftmixeddatachpt4500" --scored --evaluate  --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_sft_augmentedmixedtraingen/checkpoint-4500

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "sftmixeddatachpt4500" --scored --evaluate  --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_sft_augmentedmixedtraingen/checkpoint-4500

#CUDA_VISIBLE_DEVICES=0,1,2,3 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 python run_metric.py --dataset "attriBench" --dataset_name "subset_balanced" --split "test_ood"  --eval_model "nli_aligned" --tag  "sftmixeddata" --scored --evaluate  --model_path /lustre/fswork/projects/rech/fiz/udo61qq/Code/alignment-handbook/models/t5_xxl_true_nli_mixture_sft_augmentedmixedtraingen