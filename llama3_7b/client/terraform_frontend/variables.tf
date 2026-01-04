# variables.tf


variable "instance_type" {
  description = "Instance type for the EC2 server"
  type        = string
  default     = "g4dn.xlarge"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "ami_id" {
  description = "AMI ID for EC2 instance"
  type        = string
  # Amazon Linux 2 AMI ID - update this for your region
  default     = "ami-0cff7528ff583bf9a"
}


variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name for resource tagging"
  type        = string
  default     = "model-service"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.1.0.0/16"
}

variable "subnet_cidr" {
  description = "CIDR block for subnet"
  type        = string
  default     = "10.1.1.0/24"
}

variable "volume_size" {
  description = "Size of the root volume in GB"
  type        = number
  default     = 60
}

variable "ssh_allowed_cidr" {
  description = "CIDR block allowed for SSH access"
  type        = string
  default     = "0.0.0.0/0"  # Consider restricting this to your IP
}


variable "ssh_public_key_path" {
  description = "Path to public key for SSH access"
  type        = string
  default     = "~/.ssh/id_rsa.pub"  # Default path to public key
}



variable "docker_compose_version" {
  description = "Version of Docker Compose to install"
  default     = "v2.24.1"
}







variable "server_vpc_id" {
  type        = string
  description = "ID of the server VPC"
}


variable "instance_private_ip" {
  type        = string
  description = "ID of the server VPC"
}