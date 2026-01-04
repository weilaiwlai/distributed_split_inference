#!/bin/bash

# Define paths to the Terraform directories
BASE_DIR=$(pwd)
PEERING_DIR="./peering"
SERVER_DIR="./server"
CLIENT_DIR="./client"



# Step 1: Initialize and apply the server configuration
echo "Destroying all configurations..."
cd $BASE_DIR/$CLIENT_DIR/terraform_backend || exit
terraform destroy -lock=false -auto-approve

cd $BASE_DIR/$CLIENT_DIR/terraform_frontend || exit
terraform destroy -lock=false -auto-approve

cd $BASE_DIR/$SERVER_DIR/terraform || exit
terraform destroy -lock=false -auto-approve


cd $BASE_DIR/$PEERING_DIR || exit
terraform destroy -lock=false -auto-approve