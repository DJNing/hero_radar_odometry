docker run --gpus all --rm -it --ipc=host -p 6006:80 \
    -p 6007:22 \
    --name hero_docker \
    -v /mnt/sda1/DJ/code/hero_radar_odometry:/workspace/hero_radar_odometry \
    -v /mnt/sda1/DJ/data/under_the_radar:/workspace/under_the_radar \
    -v /mnt/sda1/DJ/data:/workspace/data\
    hero-image:latest