#!/bin/bash

# Exit on any error
set -e
BASE_DIR=$(pwd)

echo "1. Applying Terraform configuration..."
cd $BASE_DIR/terraform_backend
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

if [ "$1" = "--wait" ]; then
   echo "2. Waiting 3min for EC2 instance to be ready."
   sleep 180
else
   echo "SKIPPING 2. Waiting 3min for EC2 instance to be ready."
fi

echo "3. Copying application files to EC2..."
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa ubuntu@$EC2_IP "mkdir -p ~/app"
scp -i ~/.ssh/id_rsa ../Dockerfile ../requirements.txt ../install_model.py ubuntu@$EC2_IP:~/app/
#scp -i ~/.ssh/id_rsa -r ../model_store ubuntu@$EC2_IP:~/app/
scp -i ~/.ssh/id_rsa -r ../app ubuntu@$EC2_IP:~/app/

echo " ~/.ssh/id_rsa ubuntu@$EC2_IP "

#THIS IS SLOW AF AND HANGS/TIMEOUTS/BREAKS, need to fix.  put this in another script after the deployment.  
#ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa ubuntu@$EC2_IP << 'EOF'
#    pip install transformers==4.35.2 torch
#    cd ~/app
#     nohup  python install_model.py >  /home/ubuntu/app/model_download.log 2>&1 &#

  #  echo $! >  /home/ubuntu/app/model_download.pid#
##EOF

# 5. Monitor download progress
#echo "Model download started. Monitoring progress..."
#ssh -i ~/.ssh/id_rsa ubuntu@$EC2_IP "tail -f  /home/ubuntu/app/model_download.log" &
#TAIL_PID=$!

# 6. Wait for download to complete
#while ssh -i ~/.ssh/id_rsa ubuntu@$EC2_IP "ps -p \$(cat  /home/ubuntu/app/model_download.pid) > /dev/null 2>&1"; do
 #   echo "Download still in progress..."
  #  sleep 60
#done




echo "4. Building and pushing image on EC2..."
ssh -i ~/.ssh/id_rsa ubuntu@$EC2_IP "cd ~/app && \
    sudo aws ecr get-login-password --region us-east-1 | sudo docker login --username AWS --password-stdin $ECR_REPO && \
    sudo docker build -t model-service . && \
    sudo docker tag model-service:latest $ECR_REPO:latest && \
    sudo docker push --disable-content-trust $ECR_REPO:latest"


echo "5. Starting the service..."
PRIVATE_IP=$(terraform output -raw instance_private_ip)

ssh -i ~/.ssh/id_rsa ubuntu@$EC2_IP "
    set -e
    echo 'Creating application directory...'
    mkdir -p ~/app && cd ~/app

    echo 'Creating docker-compose.yml file...'
    echo 'version: \"3\"
services:
  model:
    image: '$ECR_REPO':latest
    restart: always1
    network_mode: host
    volumes:
      - /home/ubuntu/app/model_store:/app/model_store
    command: python main.py --master_address=$PRIVATE_IP
    runtime: nvidia
    ' > docker-compose.yml

    echo 'Pulling the latest Docker image...'
    sudo docker pull $ECR_REPO:latest || { echo 'Docker pull failed!' && exit 1; }

    echo 'Starting the service with Docker Compose...'
    sudo /usr/local/bin/docker-compose up -d || { echo 'Failed to start the service!' && exit 1; }

    echo 'Service started successfully!'
"

echo ""
echo ""

#COMMENT
terraform output




echo "Executing frontend terraform..."

# Step 2: Capture the server_vpc_id from the server configuration
instance_private_ip=$(terraform output -raw instance_private_ip)
echo $instance_private_ip=$(terraform output -raw instance_private_ip)
echo "instance_private_ip: $instance_private_ip"
cd $BASE_DIR


if grep -q "^instance_private_ip" "terraform_frontend/terraform.tfvars"; then
    # Replace the existing line
    echo "Replacing instance_private_ip in client "
    sed -i "" "s|^instance_private_ip.*|instance_private_ip = \"$instance_private_ip\"|" "terraform_frontend/terraform.tfvars"
else
    # Append the new line
    echo "Adding instance_private_ip to frontend "
    echo "instance_private_ip = \"$instance_private_ip\"" >> "terraform_frontend/terraform.tfvars"
fi



cd $BASE_DIR/terraform_frontend
terraform apply -lock=false -auto-approve 

terraform output
#exit

echo ""
echo "Client: Deployment complete!"
echo " Your service will be available at http://$EC2_IP"
echo "Note: It might take a few minutes for the service to start completely."
echo "You can SSH with: ssh -i ~/.ssh/id_rsa ubuntu@$EC2_IP"
echo "You can check the status with: ssh -i ~/.ssh/id_rsa ubuntu@$EC2_IP 'docker ps'"
echo "And view logs with: ssh -i ~/.ssh/id_rsa ubuntu@$EC2_IP 'docker-compose logs'"