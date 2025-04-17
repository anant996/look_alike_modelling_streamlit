# Use the Python 3.10 slim base image
FROM python:3.10-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port the app will run on
EXPOSE 5881

# Command to run the Streamlit app
CMD ["streamlit", "run", "landing_page.py", "--server.port", "5881"]
