




# outputs.tf
output "frontend_url" {
  value = aws_cloudfront_distribution.frontend_distribution.domain_name
}


