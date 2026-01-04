


terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"  # or whatever version you're using
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.0"
    }
  }
}


provider "aws" {
  region = "us-east-1"
}

resource "aws_key_pair" "deployer" {
  key_name   = "${var.project_name}-deployer-key"
  public_key = file(var.ssh_public_key_path)
}

# Elastic IP
resource "aws_eip" "public_ip" {
  domain   = "vpc"
  instance = aws_instance.model_server.id
}

# VPC and networking
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-vpc"
    Environment = var.environment
  }
}

resource "aws_subnet" "main" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.subnet_cidr
  availability_zone       = "${var.region}a"
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-subnet"
    Environment = var.environment
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-igw"
    Environment = var.environment
  }
}

resource "aws_route_table" "main" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  route {
    cidr_block  = data.aws_vpc.server.cidr_block
    vpc_peering_connection_id = aws_vpc_peering_connection.main.id
  }

  tags = {
    Name = "${var.project_name}-route-table"
    Environment = var.environment
  }
}





resource "aws_route_table_association" "main" {
  subnet_id      = aws_subnet.main.id
  route_table_id = aws_route_table.main.id
}

# Security group
resource "aws_security_group" "model_server" {
  name        = "${var.project_name}-sg"
  description = "Security group for model server"
  vpc_id      = aws_vpc.main.id

  // Inbound rule for ICMP traffic (ping)
  ingress {
    from_port   = -1    # -1 means all ICMP types
    to_port     = -1    # -1 means all ICMP types
    protocol    = "icmp"
    cidr_blocks = ["0.0.0.0/0"]  # Allows access from any IP address (use with caution in production)
  }
  
    ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp" 
    cidr_blocks = ["10.0.0.0/16", "10.1.0.0/16"]
    }

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }
  ingress {
    description = "HTTP"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  
  }
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_allowed_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-sg"
    Environment = var.environment
  }
}

# ECR Repository
resource "aws_ecr_repository" "model" {
  name = "${var.project_name}-repo"
  force_delete = true
  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Environment = var.environment
  }
}


# IAM Role for EC2
resource "aws_iam_role" "ec2_role" {
  name = "${var.project_name}_server_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Environment = var.environment
  }
}


data "aws_caller_identity" "current" {}
resource "aws_iam_role_policy" "ecr_policy" {
  name = "ecr_access"
  role = aws_iam_role.ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchCheckLayerAvailability",
          "ecr:ListImages",
          "ecr:DescribeImages",
          "ecr:DescribeRepositories"
        ],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = [
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ],
        Resource = "arn:aws:ecr:${var.region}:${data.aws_caller_identity.current.account_id}:repository/${var.project_name}-repo"
      }
    ]
  })
}


resource "aws_iam_instance_profile" "model_server" {
  name = "${var.project_name}_server_profile"
  role = aws_iam_role.ec2_role.name
}

# EC2 Instance
resource "aws_instance" "model_server" {


  ami           = var.ami_id
  instance_type = var.instance_type
  key_name = aws_key_pair.deployer.key_name

  subnet_id                   = aws_subnet.main.id
  vpc_security_group_ids      = [aws_security_group.model_server.id]
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.model_server.name

  root_block_device {
    volume_size = var.volume_size
    volume_type = "gp3"
  }


  user_data = <<-EOF
              #!/bin/bash
              # Update system and install dependencies
              yum update -y
              yum install -y docker git jq
              systemctl start docker
              systemctl enable docker


              # Add the ubuntu to the Docker group to allow non-sudo Docker commands
              usermod -aG docker ubuntu
              
              # Change permissions for /usr/local/bin to allow ubuntu to write
              chown -R ubuntu:ubuntu /usr/local/bin
              chmod -R u+w /usr/local/bin

              
              # Install the latest version of Docker Compose (v2.24.1)
              #DOCKER_COMPOSE_VERSION="${var.docker_compose_version}"
              DOCKER_COMPOSE_VERSION="v2.24.1"
              sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
              chmod +x /usr/local/bin/docker-compose
              
              # Verify Docker Compose installation
              /usr/local/bin/docker-compose --version || echo "Docker Compose installation failed!"

              # Create app directory
              mkdir -p /home/ubuntu/app
              
              # Configure Docker permissions
              usermod -aG docker ubuntu
              
              # Login to ECR
              aws ecr get-login-password --region ${var.region} | docker login --username AWS --password-stdin ${aws_ecr_repository.model.repository_url}
              
              # Create docker-compose.yml
              cat > /home/ubuntu/app/docker-compose.yml <<'COMPOSE'
              version: '3'
              services:
                model:
                  image: ${aws_ecr_repository.model.repository_url}:latest
                  ports:
                    - "8000:8000"
                  deploy:
                    resources:
                      limits:
                        cpus: '1.5'
                        memory: 4G
                  restart: always
              COMPOSE
              
              # Set permissions for the app directory
              chown -R ubuntu:ubuntu /home/ubuntu/app
              
              # Start the service using Docker Compose
              cd /home/ubuntu/app
              /usr/local/bin/docker-compose up -d
              EOF

  tags = {
    Name        = "${var.project_name}-server"
    Environment = var.environment
  }
}





# Then use it in data source
data "aws_vpc" "server" {
  id = var.server_vpc_id
}

# Peering connection
resource "aws_vpc_peering_connection" "main" {
  vpc_id        = aws_vpc.main.id
  peer_vpc_id   = var.server_vpc_id
  auto_accept   = true
}