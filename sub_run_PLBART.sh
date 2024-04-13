base_path=" "; # Your current path

PLBART_PATH="${base_path}/PLBART";
cd ${PLBART_PATH};
echo "#############################################################################################";
echo "Experiment for PLBART";
echo "=============================================================================================";

bash run.sh 3 my_data True 0.1

