# current_dir=$(pwd);
# base_path=$(realpath ../);
base_path="/home/weiguo/weiguo/MODIT";

CODEBERT_PATH="${base_path}/CodeBERT";
cd ${CODEBERT_PATH};
echo "#############################################################################################";
echo "Experiment for CodeBERT";
echo "=============================================================================================";
echo "My Dataset:"
echo "---------------------------------------------------------------------------------------------";

bash run.sh 3 my_data True 0.1