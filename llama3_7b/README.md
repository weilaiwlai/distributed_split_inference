## General Overview: 
There are three folders each with specific terraform.  I run them in this order: 

1. peering
2. client
3. server

We run everything in the file ```deploy_all.sh```.   Here is an overview of the logic:

1. In peering terraform create a server VPC_ID.  
2. Add the ```server_vpc_id``` to the server terraform terraform.tfvars and the client terraform.tfvars.  The client needs the ```server_vpc_id``` to create a ```vpc_peering_connection``` between the client and server. 
3. The client terraform creates a server with a peering connection and shares its private IP with the server. I share the private IP in the bash script -- in a real world implementation this would need to be done via an s3 bucket or another secure remote data sharing method.  
4. In the ```deploy_client.sh``` the client code is in the client/app folder.  I use main.py (API) and model_executor.py (model). See the Dockerfile and docker-compose.yml for additional details. 
5. In the ```deploy_client.sh``` I also deploy the frontend.  The frontend code is in the frontend folder and uses the template file index.html.tpl to fill in the private IP.  See the terraform for more details. 
6. Finally I deploy the server in the server folder ```deploy_server.sh```

The cool part of the code is in client/app/model_executor.py (```model_communication```) and server/app/model_backend.py (```generate```)


## Stuff to do: 

**Frontend**

- Add domain name to cloudfront URL
- Move conversation history processing to backend
- Improve chat UI to include code blocks for markdown etc. 

**Backend**
- Add better logging
- Add Transformer caching -- not completely sure how this works but could help immensely with generation time.  Look up "inference time caching for transformers"
- eventually transition to kubernetes so I can add other services 
- Adding autoscaling will be weird weird with multiple machines performing model inference. 


**Distributed Model Inference**
- Scale to bigger model--llama3 70b

**Split Inference, special differential privacy functionality**
- add an argument for the noise function: either regenerate noise during every next token prediction, or append noise to the existing noise (this would be a noise vector for the new word of size ```embedding_size```).  
- noise can also be set during init if we want with a specific seed. lots of options for things to do with noise. 
- a very useful piece of functionality would be a separate library for sequence/word/PII specific noise.  for example, if someone wants to obfuscate the word ```password```, then this should be handled in the library and easily configurable. if they want to obfuscate a sequence/phrase ```credit card number``` this should be handled in the library. if they want to obfuscate PII data such as ```mattg@gmail.com``` this should be handled in the library. 

**Architecture**
- IMPORTANT*** Stitch together Client/Server backends better--right now terraform first creates a server VPC for peering with client server, then creates client, then creates server, all in few scripts.  In a real world scenario I'd want to share relevant data (VPC peering connection, private IP) via AWS buckets and across remote environments, rather than running simple bash files. Need to handle this for an autoscaling scenario as well - If demand increases, how do we handle communication to scale the multiple backend servers? since we explicitly need a model-client->model-server connection, this could be tricky. this scaling logic could be significant in the success of this applicaiton.  
- Restructure code base. whole repo needs refactored. 

**Future**
- Move backend to a Trusted Execution Environment (TEE)--This would add a ton of security but could be a nightmare to implement because the NCCL communication might not be capable.  AWS has an option for this (AWS Enclave) 
- Move to different cloud providers (GCP/AZURE)
