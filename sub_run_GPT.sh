base_path=" "; # Your current path

GPT_PATH="${base_path}/GPT";
cd ${GPT_PATH};
echo "#############################################################################################";
echo "Experiment for CodeGPT";
echo "=============================================================================================";

bash run.sh 3 my_data adapted True 0.1

