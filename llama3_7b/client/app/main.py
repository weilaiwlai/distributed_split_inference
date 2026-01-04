# main.py

from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from contextlib import asynccontextmanager
import uvicorn
import argparse

# -----------------------------
# Your existing imports/modules
# -----------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from model_executor import ModelExecutor


security = HTTPBasic()
correct_username = "UStAilaN"
correct_password = "pK9#mJ4$xL2@"

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


class GenerationRequest(BaseModel):
    prompt: str
    history: Optional[str] = ""
    max_length: Optional[int] = 512
    temperature: Optional[float] = 1
    stream: Optional[bool] = False
    l2_norm: Optional[float] = 0


# -----------------------------
# Batching logic classes
# -----------------------------
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
                for f, result in zip(futures, results):
                    f.set_result(result)
            except Exception as e:
                for f in futures:
                    f.set_exception(e)


class ModelService:
    def __init__(self, args):
        self.model_executor = ModelExecutor(args)

    async def generate(self, batch: List[GenerationRequest]):
        # Batch-based inference
        responses = []
        for req in batch:
            response = self.model_executor.run(
                input_text=req.prompt,
                history=req.history,
                max_new_tokens=req.max_length,
                temperature=req.temperature,
                l2_norm=req.l2_norm,
            )
            responses.append(response)
        return responses

    async def stream_inference(self, req: GenerationRequest):
        """
        Returns an async generator that yields tokens (or chunks of text)
        as soon as they're generated.
        """
        # Simply call the async generator in ModelExecutor
        async for token_chunk in self.model_executor.stream_run(
            input_text=req.prompt,
            history=req.history,
            max_new_tokens=req.max_length,
            temperature=req.temperature,
            l2_norm=req.l2_norm
        ):
            yield token_chunk


def create_app(args, model_service):

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.model_service = model_service
        app.batch_manager = BatchManager()
        # Start the batch processing task in the background
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
        allow_origins=["https://d2lt92f6mmvv10.cloudfront.net"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------
    # Normal (batched) route
    # ---------------------
    @app.post("/generate")
    async def generate(request: GenerationRequest, username: str = Depends(authenticate)):
        # Uses the batch manager
        try:
            result_future = await app.batch_manager.add_request(request)
            return {"generated_text": await result_future}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # -------------------------
    # NEW: Streaming route
    # -------------------------
    @app.post("/generate_stream")
    async def generate_stream(request: GenerationRequest):
        async def token_generator():
            try:
                async for token in model_service.stream_inference(request):
                    yield token
            except Exception as e:
                # For unexpected errors, also just yield an error chunk or stop
                yield f"\n[CRITICAL ERROR: {str(e)}]\n"
                # Then break out, no more chunks
                return

        return StreamingResponse(token_generator(), media_type="text/plain")


    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--master_address', type=str, default="192.168.1.153")
    parser.add_argument('--master_port', type=str, default="29500")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--ifname', type=str, default="ens5")

    args = parser.parse_args()
    model_service = ModelService(args)
    fastapi_app = create_app(args, model_service)

    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="uvloop"
    )
