docker load --input graphiler.docker.tar
docker run -i -t --gpus ALL -e NVIDIA_VISIBLE_DEVICES=$NV_GPU graphiler
git pull
