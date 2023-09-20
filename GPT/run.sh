#!/usr/bin/bash

cuda_device=$1;
data_name=$2;
# source=$3;
# target=$4;
model=$3;

loc=$4;
lam=$5;

# ONLY_LOC=True;
ONLY_LOC=False;

if [[ ${data_name} = "small" ]]; then
    BLOCK_SIZE=512;
elif [[ ${data_name} = "medium" ]]; then
    BLOCK_SIZE=1024;
elif [[ ${data_name} = "my_data" ]]; then
    BLOCK_SIZE=768;
elif [[ ${data_name} = "my_data_no_line_split" ]]; then
    BLOCK_SIZE=768;
elif [[ ${data_name} = "my_data_with_location_in_src" ]]; then
    BLOCK_SIZE=768;
elif [[ ${data_name} = "my_data_no_context" ]]; then
    BLOCK_SIZE=768;
else
    echo "Data Name must be either small or medium";
    exit;
fi


if [ $data_name = my_data ];then
    max_lines=20
elif [ $data_name = my_data_with_location_in_src ];then
    max_lines=20
elif [ $data_name = my_data_no_context ];then
    max_lines=20
elif [ $data_name = my_data_no_line_split ];then
    max_lines=20
elif [ $data_name = my_data_with_line_num ];then
    max_lines=20
elif [ $data_name = small ];then
    max_lines=5
elif [ $data_name = medium ];then
    max_lines=10
fi

if [[ $model = "normal" ]]; then
    PRETRAINDIR="microsoft/CodeGPT-small-java";
    MODEL_SHAPE="normal";
elif [[ $model = "adapted" ]]; then
    PRETRAINDIR="microsoft/CodeGPT-small-java-adaptedGPT2";
    MODEL_SHAPE="adapted";
else
    echo "Model Type Must be Eith \"normal\" or \"adapted\"";
    exit;
fi

# CONFIG_DIR=${PRETRAINDIR};

# data_dir=../data/PLBART_DATA/${data_name}.${source}.${target}
data_dir=../data/${data_name}

export CUDA_VISIBLE_DEVICES=${cuda_device}
mkdir -p saved_models;
# output_dir=saved_models/${data_name}-${source}-${target}-${MODEL_SHAPE};
# output_dir=saved_models/${data_name};

# output_dir=saved_models/${data_name}_lambda_${lam};


# output_dir=saved_models/${data_name}_${model}_lambda_${lam};

output_dir=saved_models/${data_name}_${model}_smooth_gradient_step_2_bs_4_lam_${lam};

# output_dir=saved_models/${data_name}_${model}_only_loc_no_smooth_gradient_step_2_bs_4;

# output_dir=saved_models/${data_name}_${model};
# output_dir=saved_models/${data_name}_${model}_step_2_bs_4;

# output_dir=saved_models/${data_name}_${model}_only_loc_lr_5e-4_no_smooth;

mkdir -p ${output_dir};

function train(){
    CHECKPOINT_DIR=${output_dir}/checkpoint-best
    if [[ -f ${CHECKPOINT_DIR} ]]; then
        echo "Found a trained checkpoint, Not performing training!";
	      return;
    fi
    LANG=java;
    LOGFILE=${output_dir}/training.log;

    python run.py \
            --localization $loc \
            --only_loc $ONLY_LOC \
            --lam $lam \
            --max_lines $max_lines \
            --smooth_eps 0.0 \
            --data_dir=${data_dir} \
            --langs=$LANG \
            --output_dir=${output_dir} \
            --pretrain_dir=$PRETRAINDIR \
            --log_file=$LOGFILE \
            --model_type=gpt2 \
            --block_size=${BLOCK_SIZE} \
            --do_train \
            --node_index 0 \
            --learning_rate=5e-5 \
            --weight_decay=0.0 \
            --evaluate_during_training \
            --per_gpu_train_batch_size=4 \
            --per_gpu_eval_batch_size=8 \
            --gradient_accumulation_steps=2 \
            --num_train_epochs=50 \
            --logging_steps=100 \
            --overwrite_output_dir \
            --seed=42;
}

function evaluate() {
    LANG=java;
    # PRETRAINDIR=${output_dir}/checkpoint-best;
    LOGFILE=${output_dir}/evaluation.log;
    LOAD_MODEL_PATH=${output_dir}/checkpoint-best/pytorch_model.bin;

    python -u run.py \
            --localization $loc \
            --only_loc $ONLY_LOC \
            --max_lines $max_lines \
            --lam $lam \
            --data_dir=${data_dir} \
            --langs=$LANG \
            --output_dir=${output_dir} \
            --pretrain_dir=$PRETRAINDIR \
            --load_model_path=$LOAD_MODEL_PATH \
            --log_file=$LOGFILE \
            --model_type=gpt2 \
            --block_size=${BLOCK_SIZE} \
            --do_infer \
            --logging_steps=100 \
            --seed=42;
}


train;
evaluate;
