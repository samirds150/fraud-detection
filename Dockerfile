# Use official Python image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy project files
COPY . .

# Patch vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Install dependencies
RUN pip install --no-cache-dir -r requirements.prod.txt

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
