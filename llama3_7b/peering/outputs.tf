


output "server_vpc_cidr" {
  value = aws_vpc.main.cidr_block
}

output "server_vpc_id" {
  value = aws_vpc.main.id
}