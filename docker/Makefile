PROJECT ?= depth_pe
WORKSPACE ?= /workspace
DOCKER_IMAGE ?= ${PROJECT}:v0

SHMSIZE ?= 444G
WANDB_MODE ?= run
DOCKER_OPTS := \
                        --name=depth_pe \
                        -it \
                        --shm-size=${SHMSIZE} \
                        -v ~/.aws:/root/.aws \
                        -v /root/.ssh:/root/.ssh \
                        -v ~/.cache:/root/.cache \
                        -v /home/mazhuang:/home/mazhuang \
                        -v /mnt/fsx/:/mnt/fsx \
                        -v /dev/null:/dev/raw1394 \
                        -v /tmp:/tmp \
						-v /media:/media \
                        -v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0 \
                        -v /var/run/docker.sock:/var/run/docker.sock \
                        -v ${PWD}:${WORKSPACE} \
                        -w ${WORKSPACE} \
                        --privileged \
                        --ipc=host \
                        --network=host


.PHONY: all clean docker-build docker-overfit-pose

all: clean

clean:
		find . -name "*.pyc" | xargs rm -f && \
		find . -name "__pycache__" | xargs rm -rf

docker-build:
        docker build \
                -f docker/Dockerfile \
                -t ${DOCKER_IMAGE} .

docker-start-interactive:
		nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} bash

docker-start-jupyter: docker-build 
		nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
                bash -c "jupyter notebook --port=8888 -ip=0.0.0.0 --allow-root --no-browser"

docker-run: docker-build
		nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
                bash -c "${COMMAND}"

docker-run-mpi: docker-build
		nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
                bash -c "${MPI_CMD} ${COMMAND}"