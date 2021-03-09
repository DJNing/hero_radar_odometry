# HERO (Hybrid-Estimate Radar Odometry)

HERO (Hybrid-Estimate Radar Odometry) combines probabilistic state estimation and deep learned features. The network is trained in an unsupervised fashion. Deep learning is leveraged to process rich sensor data while classic state estimation is used to handle probabilities and out-of-distribution samples through outlier rejection schemes. To our knowledge, this is the first example of a totally unsupervised radar odometry pipeline. Our pipeline aproaches the performance of the current state of the art in point-based radar odometry, [Under the Radar](https://arxiv.org/abs/2001.10789), while being unsupervised.

Our implementation uses PyTorch for the deep network, and C++/[STEAM](https://github.com/utiasASRL/steam) for the state estimation back-end. We use a sliding window of length 4 to obtain the best results. Note that we train using a fixed seed (0) in order to make our training results more reproducible.

This repository also provides an implementation of Dan Barnes' [Under the Radar](https://arxiv.org/abs/2001.10789).

We trained and tested these networks on the [Oxford Radar Robotcar Dataset](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/). Results of our method compared against others is provided below. HC: Hand-crafted, L: Learned. In our paper, we also provide results on 100 km of data collected using our own platform, shown below. In addition, our paper provides results comparing the performance of radar odometry in sunny and snowy weather.

| Methods         | Supervision       | Translational Error (%) | Rotational Error (1 x 10<sup>-3</sup> deg/m) |
|-----------------|-------------------|-------------------------|:--------------------------------------------:|
| [Under the Radar](https://arxiv.org/abs/2001.10789) | Supervised (L)    | 2.0583                  | 6.7                                          |
| [RO Cen](https://www.robots.ox.ac.uk/~mobile/Papers/2018ICRA_cen.pdf)          | Unsupervised (HC) | 3.7168                  | 9.5                                          |
| [MC-RANSAC](https://arxiv.org/abs/2011.03512)       | Unsupervised (HC) | 3.3204                  | 10.95                                        |
| HERO (Ours)     | Unsupervised (L)  | 2.4076                  | 6.902                                        |

## Boreas Data-Taking Platform
![Boreas](figs/boreas.jpg "Boreas")

## Pose-Graph Architecture
![Boreas](figs/arch.jpg "Architecture")

## DNN Architecture
![Boreas](figs/dnn.jpg "DNN")

## Keypoint Covariances
![Boreas](figs/cov.jpg "Covariances")

## Odometry Performance
![Odom](figs/odom.jpg "Odom")

# Build Instrucions
We provide a Dockerfile which can be used to build a docker image with all the required dependencies installed. It is possible to build and link all required dependencies using cmake, but we do not provide instrucions for this. To use NVIDIA GPUs within docker containers, you'll need to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

## Building Docker Image:
```
cd docker
docker build -t hero-image .
```
## Launching NVIDIA-Docker Container:
```
docker run --gpus all --rm -it \
    --name hero_docker \
    -v <path_to_dataset>:/workspace/<dataset_path>  # Change to loc of oxford data
    --shm-size 16G \
    --ipc=host -p 6006:80 \  # tensorboard within docker at port 80 to 6006 on host
    hero-image:latest
```
## Building CPP-STEAM:
After launching the docker container, cpp code needs to be built:
```
mkdir cpp/build
cd cpp/build
cmake .. && make
```

# Network Configuration
`model` : HERO or UnderTheRadar \
`data_dir` : set to the parent directory of the oxford dataset sequences \
`log_dir` : write tensorboard logs to this directory \
`cart_resolution` : meters per pixel in the cartesian radar input to the DNN \
`cart_pixel_width` : width of the cartesian radar input in pixels \
`train_split` : indices of the oxford sequence used for training \
`validation_split` : index of the oxford sequence used for validation \
`test_split` : indices of the oxford sequence used for testing \
`networks[unet][first_feature_dimension]` : depth of first CNN \
`networks[keypoint][patch_size]` : width of spatial softmax patches in pixels \
`networks[softmax][softmax_temp]` : softmax temp T = (1 / (this parameter)) \
`steam[weight_matrix]` : If `true` use a 2x2 weight matrix, if `false` use a scalar weight \
`steam[log_det_thres_val]` : At test time, can be used to treshold on log(det(W)) > param \
`lr` : learning rate \
`window_size` : size of the sliding window esimator in frames (2 = normal frame-to-frame) \
`augmentation[rot_max]` : Random rotation augmentation sampled from uniform[-rot_max, rot_max]

# TODO Items

- [ ] Test cleaned up code for regression
- [ ] Add doc strings and delete unused code
- [ ] Add example usage to README
- [ ] Test UnderTheRadar implementation
- [ ] Test "test" modules
- [ ] Add script for downloading HERO results
- [ ] Train with batchsize > 2
