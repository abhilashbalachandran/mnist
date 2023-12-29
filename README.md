# Repo containing code examples for pytorch distributed training

1. Multi GPU training with torch parallel (Has reduced performance due to GIL Global interpreter lock due to the way Cpython handles multi threading in a single process). Easiest to modify from single to multi gpu (one liner)
2. Multi GPU Training with torch data distributed parallel. Better performance but requires more code changes


# To Run code for CIFAR_multigpu_kube:
You need to have kubeflow installed. I have kubeflow on a local cluster of 3 nodes with gpu.

refer to [kubeflow installation](https://www.kubeflow.org/docs/started/installing-kubeflow/)

build docker image and push to a container repository. I have used a [local docker repository](https://www.docker.com/blog/how-to-use-your-own-registry-2/)

make sure to change the image name inside the deployment yaml to the appropriate built image.

Deploy the kubernetes deployment:

```kubectl apply -f kube_deployments pytorch_job_mnist_nccl.yaml```


For more examples refer to the official set of [examples](https://github.com/kubeflow/training-operator/tree/master/examples/pytorch)
