# Use an official Python image with pip
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the project
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Set Streamlit to run without prompts
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_HOME="/app" \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"

# Start the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
