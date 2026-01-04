############
# PROVIDER #
############
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

###########
# BUCKETS #
###########
resource "aws_s3_bucket" "frontend" {
  bucket        = "model-frontend-bucket"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_website_configuration" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  index_document {
    suffix = "index.html"
  }
}

resource "aws_s3_bucket_policy" "frontend" {
  depends_on = [aws_s3_bucket_public_access_block.frontend]
  bucket     = aws_s3_bucket.frontend.id
  policy     = jsonencode({
    Version   = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadGetObject"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.frontend.arn}/*"
      },
    ]
  })
}

######################
# REMOTE STATE DATA  #
######################
data "terraform_remote_state" "client" {
  backend = "local"
  config = {
    path = "../terraform_backend/terraform.tfstate" # adjust if needed
  }
}

####################
# CLOUDFRONT DIST. #
####################
resource "aws_cloudfront_distribution" "frontend_distribution" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "Frontend Distribution"
  default_root_object = "index.html"

  # S3 origin
  origin {
    domain_name = aws_s3_bucket.frontend.bucket_regional_domain_name
    origin_id   = "S3-frontend-bucket"

    s3_origin_config {
      origin_access_identity = ""
    }
  }

  # API origin
  origin {
    #domain_name = data.terraform_remote_state.client.outputs.instance_public_ip
    domain_name="ec2-44-221-203-57.compute-1.amazonaws.com" #<-UPDATE backend script to dump this value into frontend terraform.tfvars
    origin_id   = "API-backend"

    custom_origin_config {
      http_port              = 8000
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  # Default => S3 for static
  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-frontend-bucket"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
  }

  # Route /generate => API-backend
  ordered_cache_behavior {
    path_pattern     = "/generate*"
    target_origin_id = "API-backend"

    allowed_methods  = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods   = ["GET", "HEAD"]
    forwarded_values {
      query_string = true
      cookies {
        forward = "all"
      }
      headers = ["*"]
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 0
    max_ttl                = 0
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }

  price_class = "PriceClass_100"
}

output "frontend_distribution_domain_name" {
  value = aws_cloudfront_distribution.frontend_distribution.domain_name
}

###############
# LOCAL FILES #
###############
resource "local_file" "index_html" {
  content = templatefile("../frontend/index.html.tpl", {
    server_url = "https://${aws_cloudfront_distribution.frontend_distribution.domain_name}/generate_stream"
    username   = "UStAilaN"
    password   = "pK9#mJ4$xL2@"
  })
  filename = "../frontend/index.html"
}

resource "aws_s3_object" "frontend_object" {
  bucket       = aws_s3_bucket.frontend.id
  key          = "index.html"
  content      = local_file.index_html.content
  content_type = "text/html"
}

resource "aws_s3_object" "about_page" {
  bucket       = aws_s3_bucket.frontend.id
  key          = "about.html"
  source       = "../frontend/about.html"
  content_type = "text/html"
  etag         = filemd5("../frontend/about.html")
  force_destroy = true
}

resource "aws_s3_object" "frontend_image" {
  bucket       = aws_s3_bucket.frontend.id
  key          = "image.jpg"
  source       = "../frontend/image.jpg"
  content_type = "image/jpeg"
  etag         = filemd5("../frontend/image.jpg")
}

###################
# INVALIDATION #
###################
locals {
  timestamp = timestamp()
}

resource "null_resource" "invalidate_cache" {
  triggers = {
    always_run = local.timestamp
  }

  provisioner "local-exec" {
    command = <<EOF
      aws cloudfront create-invalidation \
        --distribution-id ${aws_cloudfront_distribution.frontend_distribution.id} \
        --paths "/*"
    EOF
  }
}
