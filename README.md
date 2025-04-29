# Smart-Generative-AI-app

Smart GenAI Web Application System built using LLM, RAG, Python and Docker.

This project demonstrates a smart generative AI-based solution that integrates information from Kafka, React, and Spark folders using a Retrieval Augmented Generation (RAG) and LLM approach. The solution includes components for:

- **Data Ingestion & Preprocessing:**  
  Reads source files from specified directories, generates embeddings using HuggingFace Transformers, and indexes them with FAISS.
- **API Service:**  
  A FastAPI server that accepts natural language queries, retrieves context from the indexed documents, and simulates a generative response.
- **Spark Re-indexing Job:**  
  A sample Spark job that triggers the re-indexing process.
- **React Frontend:**  
  A simple interface for submitting queries and displaying responses.
- **Kafka Integration:**  
  A simulated Kafka consumer listens for file update events to trigger re-indexing.

## Project Structure
project/
├── api/
│   └── main.py          # FastAPI server for handling query requests
│   ├── spark/
│   │     └── reindex.py  # Spark job to trigger re-indexing
│   ├──react folder
│   ├── kafka folder
│   ├── ingestion/
│          └── ingest.py  # Ingestion: reads files, generates embeddings, and indexes using FAISS      
├── frontend/
│   ├── package.json     # React project configuration
│   └── src/
│       └── App.js       # React component for query submission and display
├── Dockerfile           # Docker configuration for API service
├── docker-compose.yml   # Docker Compose file to run API, Kafka, and Zookeeper
├── requirements.txt     # Python dependencies for the API and ingestion
└── Smart GenAI README....txt  # Instructions on how to run the project

## Prerequisites

- **Docker and Docker Compose:** Installed on your machine.
- **Node.js and npm:** For running the React frontend (unless you containerize the frontend).

## How to Run it:

### 1. Setup and Run the API Service with Docker Compose

   ```command prompt:
   cd project

### 2. Ensure the source folders (kafka folder, react folder, spark folder) exist at the appropriate paths.

### 3. Ensure you have Docker installed and up running on your device

### 4. Build and start the services:

```command prompt
docker-compose up --build
```

This builds the API service, starts Kafka and Zookeeper, and maps port 8000 for the API.

### 5. Initialize the System (REQUIRED FIRST STEP):

Once the services are running, you need to initialize the system by making a POST request to the /initialize endpoint. You can do this in one of two ways:

#### Option 1: Using cURL
```bash
curl -X POST http://localhost:8000/initialize
```

#### Option 2: Using your web browser
Visit http://localhost:8000/initialize in your web browser

You should see a success message indicating the number of documents initialized. If you see any errors, check that your documentation folders (kafka, react, spark) are properly set up in the api directory.

### 6. Start the Frontend:

```command prompt
cd frontend
npm install
npm start
```

The application will open at http://localhost:3000. You can now enter queries and the system will search through your initialized documentation.

### Troubleshooting Initialization:

If initialization fails:
1. Check that your folders exist in the correct location (api/kafka, api/react, api/spark)
2. Verify the folders contain readable text files
3. Check the API logs using:
```bash
docker-compose logs api
```
4. You can check the folder structure using:
```bash
curl http://localhost:8000/debug/folders
```

## Backend (Python-based RAG & LLM system):

```command prompt:

pip install -r requirements.txt or 

pip install kafka-python pyspark langchain transformers faiss-cpu flask pydantic torch fastapi uvicorn openai


Step 1: Install Apache Spark
follow these steps:

1. Download and Install Spark
 - Download Spark from Apache Spark Official Site.

 - Choose the latest version (e.g., Spark 3.4.1).

 - Select Pre-built for Apache Hadoop.

 - Download the .tgz file.

- Extract the Spark files

 - Extract the downloaded .tgz file to a directory (e.g., C:\spark).

Note: Inside C:\spark, you should see folders like bin, conf, and jars.

Step 2: Configure Environment Variables for Spark
1. Set SPARK_HOME and PATH
 - Press Win + R, type sysdm.cpl, and press Enter.

- Go to the Advanced tab → Click Environment Variables.

- Under System Variables, click New and add:

 . Variable name: SPARK_HOME

 . Variable value: C:\spark

- Find Path in System Variables, click Edit, and add:
 . C:\spark\bin
 . C:\spark\sbin

Click OK on all windows and restart your computer.

Step 3: Verify Spark Installation
After restarting, open Command Prompt and run:

spark-submit --version

#### Running the Spark Re-indexing Job
Ensure Apache Spark is installed on your machine.

- Navigate to the spark directory:
cd api -> cd spark

### Pass the Python Executable in the Spark job Submit Command:

spark-submit --conf spark.pyspark.python=python reindex.py


This job will re-run the ingestion process to update the FAISS index. Also, this tells Spark to use Python instead of python3.

- Navigate to the api:
```cd api

#### Run the Server: Start the FastAPI backend with:

uvicorn main:app --reload

The API will be available at http://localhost:8000.

#### Frontend

- Navigate to the frontend folder:
```cd frontend

 - Install Dependencies: install the Node packages:

npm install

- Start the Development Server: Run the React app with:

npm start

The application will open at http://localhost:3000.

