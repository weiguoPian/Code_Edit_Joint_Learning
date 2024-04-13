base_path=" "; # Your current path

GRAPHCODEBERT_PATH="${base_path}/GraphCodeBERT";
cd ${GRAPHCODEBERT_PATH};
echo "#############################################################################################";
echo "Experiment for GraphCodeBERT";
echo "=============================================================================================";
echo "My Dataset:"
echo "---------------------------------------------------------------------------------------------";

bash run.sh 3 my_data True 0.1
