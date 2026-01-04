#!/bin/bash

# Exit on any error
set -e

echo "1. Applying Terraform configuration..."
cd terraform
#terraform init
terraform apply -lock=false -auto-approve 


# Get instance IP and ECR repo
# Get instance IP and ECR repo and check if they exist
EC2_IP=$(terraform output -raw instance_public_ip)
if [ -z "$EC2_IP" ]; then
    echo "Error: EC2 IP is empty. Terraform might have failed to create the instance."
    exit 1
fi
echo "EC2 IP: $EC2_IP"

#: <<'COMMENT'
ECR_REPO=$(terraform output -raw ecr_repository_url)
if [ -z "$ECR_REPO" ]; then
    echo "Error: ECR repository URL is empty. Terraform might have failed to create the ECR repo."
    exit 1
fi
echo "ECR Repository: $ECR_REPO"


if [ "$2" = "--wait" ]; then
   echo "2. Waiting 3min for EC2 instance to be ready."
   sleep 180
else
   echo "SKIPPING 2. Waiting 3min for EC2 instance to be ready."
fi

#

echo "3. Copying application files to EC2..."
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa ubuntu@$EC2_IP "mkdir -p ~/app"
scp -i ~/.ssh/id_rsa ../Dockerfile ../requirements.txt ../install_model.py ubuntu@$EC2_IP:~/app/
scp -i ~/.ssh/id_rsa -r ../model_store ubuntu@$EC2_IP:~/app/
scp -i ~/.ssh/id_rsa -r ../app ubuntu@$EC2_IP:~/app/


#THIS IS SLOW  AND HANGS/TIMEOUTS/BREAKS, need to fix.  put this in another script after the deployment.  
#ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa ubuntu@$EC2_IP << 'EOF'
#    pip install transformers==4.35.2 torch
#    cd ~/app
#     nohup  python install_model.py > model_download.log 2>&1 &
#
#   echo $! > model_download.pid
#EOF

# 5. Monitor download progress
#echo "Model download started. Monitoring progress..."
#ssh -i ~/.ssh/id_rsa ubuntu@$EC2_IP "tail -f model_download.log" &
#TAIL_PID=$!

# 6. Wait for download to complete
#while ssh -i ~/.ssh/id_rsa ubuntu@$EC2_IP "ps -p \$(cat model_download.pid) > /dev/null 2>&1"; do
 #   echo "Download still in progress..."
 #   sleep 60
#done



echo "4. Building and pushing image on EC2..."
ssh -i ~/.ssh/id_rsa ubuntu@$EC2_IP "cd ~/app && \
    sudo aws ecr get-login-password --region us-east-1 | sudo docker login --username AWS --password-stdin $ECR_REPO && \
    sudo docker build -t model-service . && \
    sudo docker tag model-service:latest $ECR_REPO:latest && \
    sudo docker push $ECR_REPO:latest"

echo "Client IP:"
echo $1

echo "5. Starting the service..."
ssh -i ~/.ssh/id_rsa ubuntu@$EC2_IP "
    set -e
    echo 'Creating application directory...'
    mkdir -p ~/app && cd ~/app

    echo 'Creating docker-compose.yml file...'
    echo 'version: \"3\"
services:
  model:
    image: '$ECR_REPO':latest
    restart: always
    volumes:
      - /home/ubuntu/app/model_store:/app/model_store
    network_mode: host
    command: python model_backend.py --master_address=$1
    runtime: nvidia
    ' > docker-compose.yml

    echo 'Pulling the latest Docker image...'
    sudo docker pull $ECR_REPO:latest || { echo 'Docker pull failed!' && exit 1; }

    echo 'Starting the service with Docker Compose...'
    sudo /usr/local/bin/docker-compose up -d || { echo 'Failed to start the service!' && exit 1; }

    echo 'Service started successfully!'
"

#COMMENT

echo ""
echo ""
terraform output

echo ""
echo "Server: Deployment complete!"
echo "Deployment complete! Your service will be available at http://$EC2_IP"
echo "Note: It might take a few minutes for the service to start completely."
echo "You can SSH with: ssh -i ~/.ssh/id_rsa ubuntu@$EC2_IP"
echo "You can check the status with: ssh -i ~/.ssh/id_rsa ubuntu@$EC2_IP 'docker ps'"
echo "And view logs with: ssh -i ~/.ssh/id_rsa ubuntu@$EC2_IP 'docker-compose logs'"