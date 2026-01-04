# terraform.tfvars
region         = "us-east-1"

#instance_type  = "g4dn.xlarge"
instance_type  = "g5.xlarge"

environment    = "dev"
project_name   = "model-server"
volume_size    = 100
ssh_allowed_cidr = "0.0.0.0/0"  # Replace with your IP
ami_id         = "ami-0772e0bfc7cc9c5de"
server_vpc_cidr="10.0.0.0/16"
server_vpc_id = "vpc-02b83ec9d4644a35b"
vpc_peering_connection_id = "pcx-052aa0b64affc1f8d"
