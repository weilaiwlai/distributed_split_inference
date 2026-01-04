#!/bin/bash

# Exit on any error
set -e

echo "1. Applying Terraform configuration..."
cd terraform
terraform apply -auto-approve

# Get instance IP and ECR repo
# Get instance IP and ECR repo and check if they exist
EC2_IP=$(terraform output -raw instance_public_ip)
if [ -z "$EC2_IP" ]; then
    echo "Error: EC2 IP is empty. Terraform might have failed to create the instance."
    exit 1
fi
echo "EC2 IP: $EC2_IP"

ECR_REPO=$(terraform output -raw ecr_repository_url)
if [ -z "$ECR_REPO" ]; then
    echo "Error: ECR repository URL is empty. Terraform might have failed to create the ECR repo."
    exit 1
fi
echo "ECR Repository: $ECR_REPO"



echo "2. Waiting for EC2 instance to be ready for 180 seconds..."
sleep 180  # Give the instance time to initialize



echo "3. Copying application files to EC2..."
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa ec2-user@$EC2_IP "mkdir -p ~/app"
scp -i ~/.ssh/id_rsa ../Dockerfile ../requirements.txt  ec2-user@$EC2_IP:~/app/
scp -i ~/.ssh/id_rsa -r ../model_store ec2-user@$EC2_IP:~/app/
scp -i ~/.ssh/id_rsa -r ../app ec2-user@$EC2_IP:~/app/

echo "4. Building and pushing image on EC2..."
ssh -i ~/.ssh/id_rsa ec2-user@$EC2_IP "cd ~/app && \
    sudo aws ecr get-login-password --region us-east-1 | sudo docker login --username AWS --password-stdin $ECR_REPO && \
    sudo docker build -t model-service . && \
    sudo docker tag model-service:latest $ECR_REPO:latest && \
    sudo docker push $ECR_REPO:latest"

echo "5. Starting the service..."
ssh -i ~/.ssh/id_rsa ec2-user@$EC2_IP "
    set -e
    echo 'Creating application directory...'
    mkdir -p ~/app && cd ~/app

    echo 'Creating docker-compose.yml file...'
    echo 'version: \"3\"
services:
  model:
    image: '$ECR_REPO':latest
    ports:
      - \"80:8000\"
    deploy:
      resources:
        limits:
          cpus: \"1.5\"
          memory: 4G
    restart: always' > docker-compose.yml

    echo 'Pulling the latest Docker image...'
    sudo docker pull $ECR_REPO:latest || { echo 'Docker pull failed!' && exit 1; }

    echo 'Starting the service with Docker Compose...'
    sudo /usr/local/bin/docker-compose up -d || { echo 'Failed to start the service!' && exit 1; }

    echo 'Service started successfully!'
"

echo ""
echo "Deployment complete! Your service will be available at http://$EC2_IP"
echo "Note: It might take a few minutes for the service to start completely."
echo "You can check the status with: ssh -i ~/.ssh/id_rsa ec2-user@$EC2_IP 'docker ps'"
echo "And view logs with: ssh -i ~/.ssh/id_rsa ec2-user@$EC2_IP 'docker-compose logs'"