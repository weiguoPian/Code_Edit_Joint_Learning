base_path=" "; # Your current path

CODEBERT_PATH="${base_path}/CodeBERT";
cd ${CODEBERT_PATH};
echo "#############################################################################################";
echo "Experiment for CodeBERT";
echo "=============================================================================================";

bash run.sh 3 my_data True 0.1