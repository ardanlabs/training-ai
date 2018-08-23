#! /bin/bash

# docker
apt-get update
apt-get install -y docker.io

# kubeadm, kubelet, and kubectl
apt-get update && apt-get install -y apt-transport-https curl
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
cat <<EOF >/etc/apt/sources.list.d/kubernetes.list
deb http://apt.kubernetes.io/ kubernetes-xenial main
EOF
apt-get update
apt-get install -y kubelet kubeadm kubectl
apt-mark hold kubelet kubeadm kubectl

# kubeadm images
kubeadm config images pull

# add the pachrat user
sudo useradd pachrat
sudo adduser pachrat sudo

# pachctl
curl -o /tmp/pachctl.deb -L https://github.com/pachyderm/pachyderm/releases/download/v1.7.5/pachctl_1.7.5_amd64.deb && sudo dpkg -i /tmp/pachctl.deb
