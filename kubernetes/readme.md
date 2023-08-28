### Install Minikube: https://minikube.sigs.k8s.io/docs/start/

### Pushing images: https://minikube.sigs.k8s.io/docs/handbook/pushing/

### Commands to build the K8s cluster on Minikube

0. minikube start
1. alias kubectl="minikube kubectl --"
2. kubectl apply -f ./namespace/namespace.yaml
3. kubectl config set-context --current --namespace=ml_project_reference 
4. eval $(minikube docker-env) (to turn it off: eval $(minikube docker-env -u))
5. docker build -t ml_base -f ../Dockerfile_base .
6. cd .. 
7. docker build -t inference -f ./inference/Dockerfile .
8. cd ./kubernetes
7. minikube ssh -> docker images (Previously seen images are seen)
8. minikube mount /MLReferenceProject/train/artifacts/results:/home/train/artifacts/results/
9. minikube mount /MLReferenceProject/train/data/:/home/train/data/
10. kubectl apply -f ./deployments/postgres-deployment.yaml
11. kubectl apply -f ./services/postgres-service.yaml
12. kubectl apply -f ./deployments/inference-deployment.yaml
13. kubectl apply -f ./services/inference-service.yaml
14. minikube tunnel (https://minikube.sigs.k8s.io/docs/handbook/accessing/ for the LoadBalancer to have an external IP)
15. kubectl get svc (check the external IP of the load balancer)
16. minikube delete --all

### To check if the Minikube's docker daemon is used:

- env | grep DOCKER

This should display environment variables set, otherwise nothing

### Apply and remove K8s configuration

- kubectl apply -f <path_config>

- kubectl delete -f <path_config>

### Commands for debugging

- docker run -d <image> sleep infinity
- kubectl exec -it <pod_name> -c <container_name> bash
- kubectl get pods --show-labels
- kubectl config view --minify -o jsonpath='{..namespace}'
- kubectl logs
- kubectl logs --all-containers
- kubectl get events
- kubectl logs <pod_name> -c <container_name>