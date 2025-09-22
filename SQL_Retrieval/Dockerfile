# Use the official Python 3.11 image from Docker Hub
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies (optional)
COPY requirements_sql_docker.txt .
RUN pip install --no-cache-dir -r requirements_sql_docker.txt

# Copy the rest of your application code
COPY . .

# Set the default command to run your application
# Adjust 'main.py' as needed for your entrypoint
#CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "3"]
# CMD ["gunicorn", "--bind", "0.0.0.0:8001", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "api_main:app"] original had 3 workers
CMD ["gunicorn", "--bind", "0.0.0.0:8001", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "120", "api_main:app"]
