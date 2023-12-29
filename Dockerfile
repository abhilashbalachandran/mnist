FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN pip install 'git+https://github.com/lanpa/tensorboardX' && \
    pip install matplotlib

RUN mkdir -p /opt/mnist

WORKDIR /opt/mnist/src
ADD mnist.py /opt/mnist/src/mnist.py
ADD CIFAR_multinode_torchrun.py /opt/mnist/src/CIFAR_multinode_torchrun.py
ADD CIFAR_multigpu_kube.py /opt/mnist/src/CIFAR_multigpu_kube.py


RUN  chgrp -R 0 /opt/mnist \
  && chmod -R g+rwX /opt/mnist


ENTRYPOINT ["python", "/opt/mnist/src/CIFAR_multigpu_kube.py"]
