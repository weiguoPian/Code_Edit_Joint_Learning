base_path=" "; # Your current path

CODET5_PATH="${base_path}/CodeT5";
cd ${CODET5_PATH};
echo "#############################################################################################";
echo "Experiment for CodeT5";
echo "=============================================================================================";

nohup bash run.sh 3 my_data True 0.1
