#! /bin/bash

# make sure the rest of the node is ready
sleep 2m

# set kubectl config locations
echo "export KUBECONFIG=/root/.kube/config" >> /root/.bashrc
echo "export KUBECONFIG=/home/pachrat/.kube/config" >> /home/pachrat/.bashrc

# required for kube network add on
sysctl net.bridge.bridge-nf-call-iptables=1
ip="$(ifconfig | grep -A 1 'eth0' | tail -1 | cut -d ':' -f 2 | cut -d ' ' -f 1)"

# start kubernetes
kubeadm init --apiserver-advertise-address=$ip

# connect kubectl
mkdir -p $HOME/.kube
mkdir -p /home/pachrat/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
sudo cp -i /etc/kubernetes/admin.conf /home/pachrat/.kube/config
sudo chown pachrat /home/pachrat/.kube/config
export KUBECONFIG=/etc/kubernetes/admin.conf

# install networking
kubectl apply -f "https://cloud.weave.works/k8s/net?k8s-version=$(kubectl version | base64 | tr -d '\n')"

echo "Waiting for networking to come up"
start_time=$(date +%s)
while true; do
  kube_dns_running="$(kubectl get pods --all-namespaces | grep coredns | grep Running)"
  if [[ -n "$kube_dns_running" ]]; then
    break;
  fi
  printf "."
  sleep 1
  runtime=$(($(date +%s)-$start_time))
  if [ $runtime -ge 120 ]; then
    (>&2 echo "Timed out waiting for coredns (120s)")
    exit 1;
  fi
done

# master isolation
kubectl taint nodes --all node-role.kubernetes.io/master-

# allow services to act as admin (not great in general, but an easy way
# to make sure pachyderm has access to what it needs in the k8s api)
kubectl create clusterrolebinding serviceaccounts-cluster-admin \
  --clusterrole=cluster-admin \
  --group=system:serviceaccounts

# deploy pachyderm
pachctl deploy local

# set password auth
usermod -aG sudo pachrat
sudo sed -i -- 's/PasswordAuthentication no/PasswordAuthentication yes/g' /etc/ssh/sshd_config
sudo service ssh restart
