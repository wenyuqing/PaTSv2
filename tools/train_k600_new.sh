#!/usr/bin/env bash
set -x

echo 'set up environment...'
export MKL_THREADING_LAYER=GNU
GPU_PER_NODE_COUNT=`nvidia-smi -L | wc -l`
[[ -z "$AZUREML_NODE_COUNT" ]] && NODE_COUNT=1 || NODE_COUNT=$AZUREML_NODE_COUNT
[[ -z "$AZ_BATCHAI_TASK_INDEX" ]] && RANK=0 || RANK=$AZ_BATCHAI_TASK_INDEX
[[ -z "$MASTER_ADDR" ]] && MASTER_ADDR=$MASTER_IP
[[ -z "$MASTER_ADDR" ]] && MASTER_ADDR=192.168.1.30

pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements_remote.txt
pip install imgaug

#git clone https://github.com/NVIDIA/apex
#cd apex
#pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#cd ..


echo 'download data...'
SAS='?sv=2021-04-10&st=2023-01-08T13%3A06%3A43Z&se=2023-01-09T13%3A06%3A43Z&sr=c&sp=rl&sig=NS9HeqXu5X8fvP1%2BPZl0zEiDDXQPGvEvyqByXBgO5pM%3D'
wget -c https://azcopyvnext.azureedge.net/release20220721/azcopy_linux_amd64_10.16.0.tar.gz
tar -xzvf azcopy_linux_amd64_10.16.0.tar.gz

# 256 datasets
azcopy_linux_amd64_10.16.0/azcopy copy 'https://wu2train.blob.core.windows.net/v-yuqing/datasets/kinetics600/k600_data'${SAS} ./ --recursive

DATA_PATH=k600_data/rgb_videos/
TRAIN_LIST=k600_data/train.txt
VAL_LIST=k600_data/val.txt

# 416 datasets
#azcopy_linux_amd64_10.16.0/azcopy copy 'https://wu2train.blob.core.windows.net/datasets/kinetics400_416'${SAS} ./ --recursive
#DATA_PATH=kinetics400_416/
#TRAIN_LIST=kinetics400_416/train.csv
#VAL_LIST=kinetics400_416/test.csv


ln -Ts $AMLT_OUTPUT_DIR output

CONFIG=$1
PRETRAIN=$2
FRAMEWORK=$3
echo 'start running...'
python -m torch.distributed.launch --nproc_per_node=${GPU_PER_NODE_COUNT} \
                                   --node_rank=${NODE_RANK} \
                                   --nnodes=${NODE_COUNT} \
                                   --master_addr=${MASTER_ADDR} \
                                   --master_port=${MASTER_PORT} \
                                   main.py \
                                   --config $CONFIG \
                                   --backbone_path $PRETRAIN \
                                   --framework $FRAMEWORK \
                                   --output "output" \
                                   --opt \
                                   TRAIN.AUTO_RESUME True \
                                   DATA.ROOT $DATA_PATH \
                                   DATA.TRAIN_FILE $TRAIN_LIST \
                                   DATA.VAL_FILE $VAL_LIST \
                                   ${@:4}
