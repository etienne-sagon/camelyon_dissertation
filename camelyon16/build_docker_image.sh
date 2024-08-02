#!/bin/bash
#SBATCH --job-name=build_docker_image
#SBATCH --partition=its-2a30-01-part
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50GB
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --chdir=/home/ejbs1/camelyon16
#SBATCH -e /home/ejbs1/camelyon16/output/build_docker_image_%j.err
#SBATCH -o /home/ejbs1/camelyon16/output/build_docker_image_%j.out

# Start rootless Docker daemon
module load rootless-docker
start_rootless_docker.sh --quiet

# Check if Docker daemon is running
if ! pgrep -x "dockerd" > /dev/null; then
  echo "Docker daemon not running. Starting Docker daemon."
  start_rootless_docker.sh --quiet
fi

# Clean up old Docker images, containers, and volumes
echo "Cleaning up old Docker resources..."
docker system prune -a -f --volumes

# Variables
IMAGE_NAME="camelyon-image"
SCRIPT_DIR="/home/ejbs1/camelyon16/env"
RESULTS_DIR="/home/ejbs1/camelyon16/output"

# Ensure the results directory exists
mkdir -p "$RESULTS_DIR"

# Build the Docker image
DOCKER_BUILD_LOG="$RESULTS_DIR/docker_build_$(date +%s).log"
docker build -t $IMAGE_NAME -f $SCRIPT_DIR/Dockerfile $SCRIPT_DIR > "$DOCKER_BUILD_LOG" 2>&1

if [ $? -ne 0 ]; then
  echo "Docker image build failed. Check the log for details: $DOCKER_BUILD_LOG"
  exit 1
fi
