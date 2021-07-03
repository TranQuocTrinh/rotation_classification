# Build and run docker for Rotation
### Setup docker
docker build -t rotation .
### Run docker
nvidia-docker run -p 5022:5022 -t -d --restart unless-stopped --gpus all --network=host --name rotation -it rotation:latest
### Execute docker
nvidia-docker container exec -it rotation bash