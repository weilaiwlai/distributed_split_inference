
variable "project_name" {
  description = "Project name for resource tagging"
  type        = string
  default     = "model-service"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}