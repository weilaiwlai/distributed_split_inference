#!/bin/bash

# Define paths to the Terraform directories
BASE_DIR=$(pwd)
PEERING_DIR="./peering"
SERVER_DIR="./server"
CLIENT_DIR="./client"



# Step 1: Initialize and apply the server configuration
echo "Applying server configuration..."
cd $BASE_DIR/$PEERING_DIR || exit
terraform init
terraform apply -auto-approve -lock=false

# Step 2: Capture the server_vpc_id from the server configuration
SERVER_VPC_ID=$(terraform output -raw server_vpc_id)
echo "Server VPC ID: $SERVER_VPC_ID"

# Step 3: Write the server_vpc_id to the client and peering terraform.tfvars files
echo "\nWriting server_vpc_id to client and server terraform.tfvars...\n"

cd $BASE_DIR/$CLIENT_DIR/
if grep -q "^server_vpc_id" "terraform_backend/terraform.tfvars"; then
    # Replace the existing line
    echo "Replacing server_vpc_id in client "
    sed -i "" "s|^server_vpc_id.*|server_vpc_id = \"$SERVER_VPC_ID\"|" "terraform_backend/terraform.tfvars"
else
    # Append the new line
     echo "Adding server_vpc_id to client "
    echo "server_vpc_id = \"$SERVER_VPC_ID\"" >> "terraform_backend/terraform.tfvars"
fi

cd $BASE_DIR/$SERVER_DIR/
if grep -q "^server_vpc_id" "terraform/terraform.tfvars"; then
    # Replace the existing line
    echo "Replacing server_vpc_id in server "
    sed -i "" "s|^server_vpc_id.*|server_vpc_id = \"$SERVER_VPC_ID\"|" "terraform/terraform.tfvars"
else
    # Append the new line
    echo "Adding server_vpc_id to server "
    echo "server_vpc_id = \"$SERVER_VPC_ID\"" >> "terraform/terraform.tfvars"
fi





# Step 4: Initialize and apply the client configuration
echo "Applying client configuration..."

cd $BASE_DIR/$CLIENT_DIR
#terraform apply -lock=false -auto-approve 
if [ "$1" = "--wait" ]; then
   sh deploy_client.sh  --wait
else
   sh deploy_client.sh  #--wait
fi




cd $BASE_DIR/$CLIENT_DIR/terraform_backend
# Step 5: Capture the vpc_peering_connection_id from the client configuration
VPC_PEERING_CONNECTION_ID=$(terraform output -raw vpc_peering_connection_id)
echo "VPC Peering Connection ID: $VPC_PEERING_CONNECTION_ID"


CLIENT_PRIVATE_IP=$(terraform output -raw instance_private_ip)
echo "CLIENT_PRIVATE_IP: $VPC_PEERING_CONNECTION_ID"

cd $BASE_DIR/$SERVER_DIR/


# Step 6: Write the vpc_peering_connection_id to the server terraform.tfvars file
echo "Writing vpc_peering_connection_id to server terraform.tfvars..."
if grep -q "^vpc_peering_connection_id" "terraform/terraform.tfvars"; then
    # Replace the existing line
    echo "Replacing server_vpc_id in client "
    sed -i "" "s|^vpc_peering_connection_id.*|vpc_peering_connection_id = \"$VPC_PEERING_CONNECTION_ID\"|" "terraform/terraform.tfvars"
else
    # Append the new line
    echo "Adding server_vpc_id to client "
    echo "vpc_peering_connection_id = \"$VPC_PEERING_CONNECTION_ID\"" >> "terraform/terraform.tfvars"
fi

#echo "sleeping"
##sleep 45

# Step 7: Initialize and apply the server configuration
echo "Applying server configuration..."



#cd terraform
#terraform apply -lock=false -auto-approve 
if [ "$1" = "--wait" ]; then
   sh deploy_server.sh  $CLIENT_PRIVATE_IP --wait 
else
   sh deploy_server.sh   $CLIENT_PRIVATE_IP
fi




echo "Deployment complete!"

echo "\Peering output:"
cd $BASE_DIR/$PEERING_DIR
terraform output

echo "\n\nClient output:"
cd $BASE_DIR/$CLIENT_DIR/terraform_backend
terraform output

echo "Client frontend:"
cd $BASE_DIR/$CLIENT_DIR/terraform_frontend
terraform output

echo "\nServer output:"
cd $BASE_DIR/$SERVER_DIR/terraform
terraform output
