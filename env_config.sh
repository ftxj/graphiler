docker load --input graphiler-dgl.tar
docker run -i -t --gpus all -e NVIDIA_VISIBLE_DEVICES=$NV_GPU graphiler-dgl
git pull
