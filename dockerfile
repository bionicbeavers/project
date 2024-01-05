# Use the Python 3.10 slim base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to the working directory in the container
COPY . .

# Install the Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Define the default command to run when the container starts
CMD ["python", "app.py"]
