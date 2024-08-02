#!/bin/bash
#SBATCH --job-name=run_job
#SBATCH --partition=its-2a30-01-part
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50GB
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/ejbs1/camelyon16
#SBATCH -e /home/ejbs1/camelyon16/output/run_job_%j.err
#SBATCH -o /home/ejbs1/camelyon16/output/run_job_%j.out

# Start rootless Docker daemon
module load rootless-docker
start_rootless_docker.sh --quiet

# Ensure Docker daemon is running
if ! pgrep -x "dockerd" > /dev/null; then
  echo "Docker daemon not running. Starting Docker daemon."
  start_rootless_docker.sh --quiet
fi

# Variables
IMAGE_NAME="camelyon-image"
SCRIPT_DIR="/home/ejbs1/camelyon16/env"
MAIN_SCRIPT_PATH="/home/ejbs1/camelyon16/main.py"
RESULTS_DIR="/home/ejbs1/camelyon16/output"

# Ensure the results directory exists
mkdir -p "$RESULTS_DIR"
# Ensure the patch directory exists
mkdir -p "/home/ejbs1/camelyon16/patch_classification/train/normal"
mkdir -p "/home/ejbs1/camelyon16/patch_classification/train/tumor"
mkdir -p "/home/ejbs1/camelyon16/patch_classification/val/normal"
mkdir -p "/home/ejbs1/camelyon16/patch_classification/val/tumor"

mkdir -p "/home/ejbs1/camelyon16/models/"


# Run the Docker container and mount the script directory and results directory
docker run --gpus all --shm-size=16g --rm \
  -v "/home/shared/camelyon16:/app/data" \
  -v "/home/ejbs1/camelyon16:/app/scripts" \
  -w /app/scripts \
  $IMAGE_NAME python3 main.py > "$RESULTS_DIR/main_output_$(date +%s).log" 2>&1


if [ $? -ne 0 ]; then
  echo "Script execution failed. Check the output log for details: $RESULTS_DIR/main_output_$(date +%s).log"
  exit 1
fi
