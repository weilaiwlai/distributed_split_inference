FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ .
#COPY model_store ./model_store


# Add install.py to the container
#COPY install_model.py .
#RUN python install.py


ENV PYTHONUNBUFFERED=1 
ENV TRANSFORMERS_CACHE=/app/model_store
ENV PORT=8000

#COPY install_model.py .
#RUN python install_model.py


EXPOSE 8000
EXPOSE 29500

CMD ["bash"]
