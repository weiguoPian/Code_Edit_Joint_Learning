cuda_device=$1;
data_name=$2;
# source=$3;
# target=$4;
# model=$5;

loc=$3
lam=$4

ONLY_LOC=False;

source_length=512;
target_length=256;

if [ $data_name = my_data ];then
    max_lines=20
elif [ $data_name = my_data_with_location_in_src ];then
    max_lines=20
elif [ $data_name = my_data_no_line_split ];then
    max_lines=20
elif [ $data_name = my_data_no_context ];then
    max_lines=20
elif [ $data_name = my_data_with_line_num ];then
    max_lines=20
elif [ $data_name = small ];then
    max_lines=5
elif [ $data_name = medium ];then
    max_lines=10
fi


PRETRAINDIR="Salesforce/codet5-base";

# if [[ $model = "normal" ]]; then
#     PRETRAINDIR="uclanlp/plbart-base";
#     MODEL_SHAPE="normal";
# elif [[ $model = "adapted" ]]; then
#     PRETRAINDIR="microsoft/CodeGPT-small-java-adaptedGPT2";
#     MODEL_SHAPE="adapted";
# else
#     echo "Model Type Must be Eith \"normal\" or \"adapted\"";
#     exit;
# fi

# data_dir=../data/PLBART_DATA/${data_name}.${source}.${target}

# data_dir=../data/${data_name}
data_dir=../data/${data_name}

export CUDA_VISIBLE_DEVICES=${cuda_device}
mkdir -p saved_models;

output_dir=saved_models/${data_name}_step_4_bs_8;
# output_dir=saved_models/${data_name}_only_loc_step_4_bs_8_no_smooth;


mkdir -p ${output_dir};

function train(){
    CHECKPOINT_DIR=${output_dir}/checkpoint-best
    if [[ -f ${CHECKPOINT_DIR} ]]; then
        echo "Found a trained checkpoint, Not performing training!";
	      return;
    fi
    LOGFILE=${output_dir}/training.log;

    train_file=$data_dir/train_src,$data_dir/train_tgt,$data_dir/train_location;
    dev_file=$data_dir/eval_src,$data_dir/eval_tgt,$data_dir/eval_location;

    # train_file=$data_dir/train_src,$data_dir/train_tgt,$data_dir/train_location;
    # dev_file=$data_dir/eval_src,$data_dir/eval_tgt,$data_dir/eval_location;

    python run_codet5.py \
            --localization $loc \
            --only_loc $ONLY_LOC \
            --lam $lam \
            --max_lines $max_lines \
            --smooth_eps 0.1 \
            --output_dir=${output_dir} \
            --train_file=${train_file} \
            --dev_file=${dev_file} \
            --pretrain_dir=$PRETRAINDIR \
            --log_file=$LOGFILE \
            --model_type=CodeT5 \
            --source_length=${source_length} \
            --target_length=${target_length} \
            --do_train \
            --node_index 0 \
            --learning_rate=5e-5 \
            --weight_decay=0.0 \
            --evaluate_during_training \
            --per_gpu_train_batch_size=8 \
            --per_gpu_eval_batch_size=8 \
            --gradient_accumulation_steps=4 \
            --num_train_epochs=50 \
            --logging_steps=100 \
            --overwrite_output_dir \
            --seed=1234;
}

function evaluate() {
    # PRETRAINDIR=${output_dir}/checkpoint-best
    LOGFILE=${output_dir}/evaluation.log

    dev_file=$data_dir/eval_src,$data_dir/eval_tgt,$data_dir/eval_location;
    test_file=$data_dir/test_src,$data_dir/test_tgt,$data_dir/test_location;

    LOAD_MODEL_PATH=${output_dir}/checkpoint-best/pytorch_model.bin;

    python -u run_codet5.py \
            --load_model_path=$LOAD_MODEL_PATH \
            --localization $loc \
            --only_loc $ONLY_LOC \
            --lam $lam \
            --max_lines $max_lines \
            --output_dir=${output_dir} \
            --dev_file=${dev_file} \
            --test_file=${test_file} \
            --pretrain_dir=$PRETRAINDIR \
            --log_file=$LOGFILE \
            --model_type=CodeT5 \
            --source_length=${source_length} \
            --target_length=${target_length} \
            --do_infer \
            --logging_steps=100 \
            --seed=1234;
}

train;
evaluate;
