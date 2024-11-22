#!/bin/bash

# Environment variables associated with XPK on GCP.
export ZONE=...
export PROJECT=...
export TPU_TYPE=v6e-256
export NUM_SLICE=1
export CLUSTER_NAME=xpk-$USER-... # use existing CLUSTER if you have

# Environment variables associated with training config.
export BATCH_PER_DEVICE=4
export SEQUENCE_LENGTH=4096
export MAX_STEP=50
export WORKLOAD_NAME=${USER}-xpk-${TPU_TYPE}-... # Your workload name. Need to update for different run.
export BASE_DOCKER_IMAGE=us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-tpu-mixtral:v1
export PROFILE_LOG_DIR=... # GCS bucket to store profile in form of gs://...
export HF_TOKEN=... # Add your own Hugging face token to download model
