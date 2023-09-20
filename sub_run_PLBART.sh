# current_dir=$(pwd);
# base_path=$(realpath ../);
base_path="/home/weiguo/weiguo/MODIT";

GRAPH_CODEBERT_PATH="${base_path}/PLBART";
cd ${GRAPH_CODEBERT_PATH};
echo "#############################################################################################";
echo "Experiment for PLBART";
echo "=============================================================================================";
echo "My Dataset:"
echo "---------------------------------------------------------------------------------------------";

bash run.sh 3 my_data True 0.1

