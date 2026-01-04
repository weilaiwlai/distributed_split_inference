# outputs.tf
output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.model.repository_url
}

output "instance_public_ip" {
  description = "Public IP of the EC2 instance"
  value       = aws_instance.model_server.public_ip
}

output "instance_public_dns" {
  description = "Public DNS of the EC2 instance"
  value       = aws_instance.model_server.public_dns
}

output "ssh_command" {
  description = "Command to SSH into the instance"
  value       = "ssh ec2-user@${aws_instance.model_server.public_ip}"
}

output "region" {
  value = var.region
}

output "instance_id" {
  value = aws_instance.model_server.id
}


output "instance_private_ip" {
  value = aws_instance.model_server.private_ip
}

output "vpc_peering_connection_id" {
  value = aws_vpc_peering_connection.main.id
}

