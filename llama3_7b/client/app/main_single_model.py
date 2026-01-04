from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from contextlib import asynccontextmanager
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

cache_dir = "./cache/"

class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ModelService:
    def __init__(self):
        # Use the full model name from HuggingFace
        model_name = "gpt2"  # or "openai-community/gpt2"
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True, 
            cache_dir=cache_dir
        )
        
        if torch.cuda.is_available():
            print("Moving model to GPU...")
            self.model = self.model.cuda()
        print("Model loading complete!")

    async def generate(self, batch: List[GenerationRequest]):
        # Process batch of requests
        prompts = [req.prompt for req in batch]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_length=batch[0].max_length,
                temperature=batch[0].temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True
            )
        
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return responses
class BatchManager:
    def __init__(self, max_batch_size=32, max_wait_time=0.1):
        self.queue = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time

    async def add_request(self, request: GenerationRequest):
        future = asyncio.Future()
        await self.queue.put((request, future))
        return future

    async def process_batches(self, inference_fn):
        while True:
            batch = []
            futures = []

            # Get first request
            request, future = await self.queue.get()
            batch.append(request)
            futures.append(future)

            # Accumulate batch
            timeout = asyncio.create_task(asyncio.sleep(self.max_wait_time))
            
            while len(batch) < self.max_batch_size and not timeout.done():
                try:
                    request, future = await asyncio.wait_for(
                        self.queue.get(), 
                        timeout=self.max_wait_time
                    )
                    batch.append(request)
                    futures.append(future)
                except asyncio.TimeoutError:
                    break

            # Process batch
            try:
                results = await inference_fn(batch)
                for future, result in zip(futures, results):
                    future.set_result(result)
            except Exception as e:
                for future in futures:
                    future.set_exception(e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize model and batch manager
    app.model_service = ModelService()
    app.batch_manager = BatchManager()
    app.inference_task = asyncio.create_task(
        app.batch_manager.process_batches(app.model_service.generate)
    )
    yield
    # Cleanup
    app.inference_task.cancel()
    try:
        await app.inference_task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate(request: GenerationRequest):
    try:
        result = await app.batch_manager.add_request(request)
        return {"generated_text": await result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="uvloop"
    )