# Build and run docker for Rotation
### Setup docker
```sh
docker build -t rotation .
```
### Run docker
```sh
nvidia-docker run -p 5022:5022 -t -d --restart unless-stopped --gpus all --network=host --name rotation -it rotation:latest
```
### Execute docker
```sh
nvidia-docker container exec -it rotation bash
```
