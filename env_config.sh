docker load --input graphiler.docker.tar
docker run -i -t --gpus all -e NVIDIA_VISIBLE_DEVICES=$NV_GPU graphiler
git pull
