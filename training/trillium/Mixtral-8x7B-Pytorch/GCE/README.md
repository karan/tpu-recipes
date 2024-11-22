# Instructions for training Mixtral-8X7B on Trillium(v6e) TPU


This user guide provides a concise overview of the essential steps required to run HuggingFace (HF) Mixtral training on Cloud TPUs.


## Environment Setup

Please follow the corresponding TPU generation's user guide to setup the GCE TPUs
first.

Please replace all your-* with your TPUs' information.

```
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project
```

You may use this command to create a 256 chip v6e slice:

```
gcloud alpha compute tpus tpu-vm create $TPU_NAME \
    --accelerator-type v6e-256 --project $PROJECT --zone $ZONE \
    --version v2-alpha-tpuv6e
```

## Steps to Run HF Mixtral 8x7B

The following setup runs the training job with Mixtral 8x7B on GCE TPUs using the docker image from this registry (``), the docker image uses the pytorch and torch_xla nightly build from 10/28/2024 and installed with all the package dependency needed to run the model training. All the command below should run from your own machine (not the TPU host you created).

1. git clone and navigate to this README repo and run training script:
```bash
git clone https://github.com/AI-Hypercomputer/tpu-recipes.git
cd training/trillium/Mixtral-8x7B-PyTorch
```
2. Edit `env.sh` to add the hugging face token and/or setup the training parameters.
```bash
# add your hugging face token
HF_TOKEN=hf_***
```
3. Edit `host.sh` to add the docker image URL if default docker image is not accessible to you.
```bash
# docker image URL to use for the training
DOCKER_IMAGE=us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-tpu-mixtral:v0
```
4. Run the training script:
```bash
./benchmark.sh
```
`benchmark.sh` script will upload 1) environment parameters in `env.sh`, 2) model related config in `config.json`, `fsdp_config.json`, 3) docker launch script in `host.sh` and 4) python training command in `train.sh` into all TPU workers, and starts the training afterwards. When all training steps complete, it will print out training metrics of each worker as below in terminal:
```
***** train metrics *****
[worker :3] ***** train metrics *****
[worker :3]   epoch                    =      0.0391
[worker :3]   total_flos               = 216428520GF
[worker :3]   train_loss               =       8.443
[worker :3]   train_runtime            =  0:04:23.15
[worker :3]   train_samples            =       32816
[worker :3]   train_samples_per_second =       4.864
```
In addition,  it will copy back the trained model under `output/*`.