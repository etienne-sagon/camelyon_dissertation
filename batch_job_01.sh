#!/bin/bash
#SBATCH --job-name=run_docker_job
#SBATCH --partition=its-2a30-01-part
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50GB
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/ejbs1/my_docker_test
#SBATCH -e /home/ejbs1/my_docker_test/output/run_docker_job_%j.err
#SBATCH -o /home/ejbs1/my_docker_test/output/run_docker_job_%j.out

# Start rootless Docker daemon
module load rootless-docker
start_rootless_docker.sh --quiet

# Ensure Docker daemon is running
if ! pgrep -x "dockerd" > /dev/null; then
  echo "Docker daemon not running. Starting Docker daemon."
  start_rootless_docker.sh --quiet
fi

# Remove old Docker images, containers, and volumes
echo "Cleaning up old Docker resources..."
docker system prune -a -f --volumes

# Variables
IMAGE_NAME="my-custom-image"
SCRIPT_DIR="/home/ejbs1/my_docker_test"
MAIN_SCRIPT_PATH="$SCRIPT_DIR/hello.py"
RESULTS_DIR="$SCRIPT_DIR/results"

# Ensure the results directory exists
mkdir -p "$RESULTS_DIR"

# Build the Docker image
DOCKER_BUILD_LOG="$RESULTS_DIR/docker_build_$(date +%s).log"
docker build -t $IMAGE_NAME -f $SCRIPT_DIR/Dockerfile $SCRIPT_DIR > "$DOCKER_BUILD_LOG" 2>&1

if [ $? -ne 0 ]; then
  echo "Docker image build failed. Check the log for details: $DOCKER_BUILD_LOG"
  exit 1
fi

# Run the Docker container and mount the script directory and results directory
docker run --gpus all --shm-size=16g --rm -v "$SCRIPT_DIR:/app" -w /app $IMAGE_NAME python3 hello.py > "$RESULTS_DIR/hello_output_$(date +%s).log" 2>&1

if [ $? -ne 0 ]; then
  echo "Script execution failed. Check the output log for details: $RESULTS_DIR/hello_output_$(date +%s).log"
  exit 1
fi
