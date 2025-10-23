# Use the official Python 3.11 image from Docker Hub
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies (optional)
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy the rest of your application code
COPY . .


# Set the default command to run your application
# Adjust 'main.py' as needed for your entrypoint
#CMD ["nohup", "uvicorn", "cpi_top5_results_v6_vm_experimental_citeurl:app", "--host", "0.0.0.0", "--port", "8787", "--workers", "2", ">", "output.log", "2>&1", "&"]
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "deep_insights_v8_retrieval:app"]