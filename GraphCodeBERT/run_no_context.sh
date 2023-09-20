cuda_device=$1;
data_name=$2;
# source=$3;
# target=$4;
loc=$3;
lam=$4;
# max_lines=$4;

if [ $data_name = my_data ];then
    max_lines=20
elif [ $data_name = my_data_with_location_in_src ];then
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

lr=5e-5;
# lr=5e-4;

batch_size=16;
beam_size=5;
source_length=512;
# source_length=1024;
target_length=256;

data_dir=../data/${data_name}_no_context
# data_dir=../data/${data_name}_with_location_in_src


export CUDA_VISIBLE_DEVICES=${cuda_device}
mkdir -p saved_models;

# output_dir=saved_models/${data_name}_lambda_${lam}_18;

output_dir=saved_models/my_data_no_context;

# output_dir=saved_models/my_data_with_location_in_src_True_lambda_0.0;


mkdir -p ${output_dir};

function train () {
    best_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin;
    if [[ -f ${best_model} ]]; then
	    echo "Found a trained checkpoint, Not performing training!";
	    return;
	  fi
    train_file=$data_dir/train_src,$data_dir/train_tgt,$data_dir/train_location
    dev_file=$data_dir/eval_src,$data_dir/eval_tgt,$data_dir/eval_location
    eval_steps=5000
    train_steps=200000
    # train_steps=90000
    pretrained_model="microsoft/graphcodebert-base";
    python run.py \
        --do_train --do_eval \
        --localization $loc \
        --lam $lam \
        --max_lines $max_lines \
        --smooth_eps 0.1 \
        --model_type roberta --config_name roberta-base --tokenizer_name roberta-base \
        --model_name_or_path $pretrained_model \
        --train_filename $train_file --dev_filename $dev_file \
        --output_dir $output_dir \
        --max_source_length $source_length --max_target_length $target_length \
        --beam_size $beam_size \
        --train_batch_size $batch_size --eval_batch_size 8 --gradient_accumulation_steps 2 \
        --learning_rate $lr --train_steps $train_steps --eval_steps $eval_steps;
}


function evaluate () {
    dev_file=$data_dir/eval_src,$data_dir/eval_tgt,$data_dir/eval_location
    test_file=$data_dir/test_src,$data_dir/test_tgt,$data_dir/test_location
    test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
    OUTPUT_FILE=${output_dir}/evaluation_result.txt;
    python run.py \
        --do_test \
        --localization $loc \
        --max_lines $max_lines \
        --lam $lam \
        --model_type roberta --model_name_or_path roberta-base --config_name roberta-base \
        --tokenizer_name roberta-base  --load_model_path $test_model \
        --dev_filename $dev_file --test_filename $test_file \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --eval_batch_size 16 | tee $OUTPUT_FILE
}


train;
evaluate;