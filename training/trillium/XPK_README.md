## Initialization

> **_NOTE:_** We recommend running these instructions and kicking off your recipe 
workloads from a VM in GCP using Python 3.10.

1. Run the following commands to initialize the project and zone.
```shell
export PROJECT=#<your_project_id>
export ZONE=#<zone>
gcloud config set project $PROJECT
gcloud config set compute/zone $ZONE
```

2. Install XPK by following the [prerequisites](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#prerequisites) and [installation](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#installation) 
instructions. Also ensure you have the proper [GCP permissions](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#installation).

* In order to run the tpu-recipes as-is, run the `git clone` command from your home (~/) directory:
```shell
# tpu-recipes requiring XPK will look for it in the home directory
cd ~/
git clone https://github.com/google/xpk.git
```

3. Run the rest of these commands from the cloned XPK directory:

```shell
cd xpk # Should be equivalent to cd ~/xpk
```

> **_NOTE:_** If you use a virtual environment in the 
[XPK Installation](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#installation)
steps, you must use the same one to run the steps in the [MAXTEXT_README](MAXTEXT_README.md)
as well as your relevant tpu-recipe workloads.

## GKE Cluster Creation 
1. Specify your TPU GKE cluster configs.
```shell
export CLUSTER_NAME=v6e-demo #<your_cluster_name>
export NETWORK_NAME=${CLUSTER_NAME}-only-mtu9k
export NETWORK_FW_NAME=${NETWORK_NAME}-only-fw
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"
export TPU_TYPE=v6e-256 #<your TPU Type>
export NUM_SLICES=1 #<number of TPU node-pools you want to create>
export ZONE=<compute_zone>
export REGION=<compute_region>
```

2. Create the network and firewall for this cluster if it doesn’t exist yet.
```shell
NETWORK_NAME_1=${CLUSTER_NAME}-mtu9k-1-${ZONE}
NETWORK_FW_NAME_1=${NETWORK_NAME_1}-fw-1-${ZONE}

# Use a custom network for better performance as well as avoid the default network to be overloaded.
gcloud compute networks create ${NETWORK_NAME_1} --mtu=8896 --project=${PROJECT} --subnet-mode=auto --bgp-routing-mode=regional
gcloud compute firewall-rules create ${NETWORK_FW_NAME_1} --network ${NETWORK_NAME_1} --allow tcp,icmp,udp --project=${PROJECT}

# Secondary subnet for multinic experience. Need custom ip routing to be different from first network’s subnet.
export NETWORK_NAME_2=${CLUSTER_NAME}-privatenetwork-2-${ZONE}
export SUBNET_NAME_2=${CLUSTER_NAME}-privatesubnet-2-${ZONE}
export FIREWALL_RULE_NAME=${CLUSTER_NAME}-privatefirewall-2-${ZONE}
export ROUTER_NAME=${CLUSTER_NAME}-network-2-${ZONE}
export NAT_CONFIG=${CLUSTER_NAME}-natconfig-2-${ZONE}

gcloud compute networks create "${NETWORK_NAME_2}" --mtu=8896 --bgp-routing-mode=regional --subnet-mode=custom --project=$PROJECT
gcloud compute networks subnets create "${SUBNET_NAME_2}" --network="${NETWORK_NAME_2}" --range=10.10.0.0/18 --region="${REGION}" --project=$PROJECT
gcloud compute firewall-rules create "${FIREWALL_RULE_NAME}" --network "${NETWORK_NAME_2}" --allow tcp,icmp,udp --project="${PROJECT}"
gcloud compute routers create "${ROUTER_NAME}" \
  --project="${PROJECT}" \
  --network="${NETWORK_NAME_2}" \
  --region="${REGION}"
gcloud compute routers nats create "${NAT_CONFIG}" \
  --router="${ROUTER_NAME}" \
  --region="${REGION}" \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges \
  --project="${PROJECT}" \
  --enable-logging
```

3. Create GKE cluster with TPU node-pools
```shell
export CLUSTER_ARGUMENTS="--enable-dataplane-v2 --enable-ip-alias --enable-multi-networking --network=${NETWORK_NAME_1} --subnetwork=${SUBNET_NAME_1}"

export NODE_POOL_ARGUMENTS="--additional-node-network network=${NETWORK_NAME_2},subnetwork=${SUBNET_NAME_2}"

python3 xpk.py cluster create --cluster $CLUSTER_NAME --cluster-cpu-machine-type=n1-standard-8 --num-slices=$NUM_SLICES --tpu-type=$TPU_TYPE --zone=$ZONE  --project=$PROJECT --on-demand --custom-cluster-arguments="${CLUSTER_ARGUMENTS}" --custom-nodepool-arguments="${NODE_POOL_ARGUMENTS}" --create-vertex-tensorboard
```

  * Noted: TPU has `reserved`, `on-demand`, `spot` quota. This example used the `on-demand` quota. If you have the reserved or spot quota, please refer to this [link](https://github.com/google/xpk?tab=readme-ov-file#cluster-create).
  * If you want to check what quota you have, please refer to this [link](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus#ensure-quota).
  * You should be able to see your GKE cluster similar to this once it is created successfully:![image](https://github.com/user-attachments/assets/60743411-5ee5-4391-bb0e-7ffba4d91c1d)

4. Performance Daemonset 
```shell
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/ai-on-gke/9ff340f07f70be0130454f9e7238551587242b75/scripts/network-setup/v6e-network-optimization.yaml
```

5. Test your GKE cluster to make sure it is usable
```shell
python3 xpk.py workload create \
--cluster ${CLUSTER_NAME} \
--workload hello-world-test \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--command "echo Hello World"
```
* You should be able to to see results like this: ![image](https://github.com/user-attachments/assets/c33010a6-e109-411e-8fb5-afb4edb3fa72)

6. You can also check your workload status with the following command:
```shell
python3 xpk.py workload list --cluster ${CLUSTER_NAME}
```
7. For more information about XPK, please refer to this [link](https://github.com/google/xpk).

## GKE Cluster Deletion
You can use the following command to delete GKE cluster:
```shell
export CLUSTER_NAME=v6e-demo #<your_cluster_name>

python3 xpk.py cluster delete --cluster $CLUSTER_NAME
```