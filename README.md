# CVMatching
Project Semester 5 - Groep **Weekend**

## About The Project
CVMatching is a project where we use NLP to automatically analyze and match resumes with job descriptions. This repository contains the full pipeline, from data preprocessing and model training to a vector database and an interactive Streamlit application.

The project consists of three main services managed by Docker:
- **Streamlit App**: The interactive web interface for ranking resumes.

- **Qdrant Database**: The vector database that stores and searches job/resume embeddings.

- **Doccano**: A data annotation tool used for creating NER training data.

## Table of Contents

- [Recommended Setup (Docker-Compose)](#recommended-setup-docker-compose)

- [Manual Setup (Local Development)](#manual-setup-local-development)

- [Advanced Usage & Configuration](#advanced-usage--configuration)

- [Project Structure](#project-structure)

## Recommended Setup (Docker-Compose)

This is the easiest and most reliable way to run the entire project. It will build the app and launch all three services (Streamlit, Qdrant, Doccano) at once.

### Prerequisites

- [Git](https://git-scm.com/install/)
- [Docker](https://docs.docker.com/get-started/get-docker/) (Docker Desktop is recommended as it includes `docker-compose`).

### 1. Clone Repository
```bash
git clone https://github.com/XavierKdG/CVMatching.git
cd CVMatching
```

### 2. Add Dataset
download the datasets below and place the it in the `data/raw/` directory

- [Resume.csv](https://www.kaggle.com/datasets/014ce6313a60bf1563ff9ef3d57879bd8e7c1e1be0e8926bffb82d51ee85fda8?select=Resume.csv)

- [job_descriptions2.csv](https://www.kaggle.com/datasets/014ce6313a60bf1563ff9ef3d57879bd8e7c1e1be0e8926bffb82d51ee85fda8?select=job_descriptions2.csv)

### 3. Build and Run All Services
This command will build your custom Streamlit app image and start all three services. This may take several minutes on the first run.

```bash
docker-compose up --build
```

*(Note: The configs/config.yml is already pre-configured to work with this Docker setup. The app will connect to http://qdrant:6333.)*

### 4. Run the Data Pipeline (One-Time Setup)
Your app is running, but the Qdrant database is **empty**. You must run your Python pipeline inside the app container to process your data and fill the database.

1. Open a **new, separate terminal** (leave `docker-compose up` running).

2. Execute a shell inside the running `app` container:

3. You are now inside the container. Run your full pipeline:
```bash
python pipeline.py
```
This will run `preprocess.py`, `train.py`, and `upload_to_qdrant.py`, populating your Qdrant database.

### 5. Access Your Services

- Streamlit App: `http://localhost:8501`

- Doccano UI: `http://localhost:8000` (Login: `admin` / `password`)

- Qdrant Web UI: `http://localhost:6334`

### Stopping the Services

To stop all services, press `Ctrl+C` in the terminal where `docker-compose up` is running, or run this command from the project directory in another terminal:

```bash
docker-compose down
```

## Manual Setup (Local Development)
Follow these steps if you want to run the project locally without using docker-compose for the Streamlit app.

### 1. Clone Repository

```bash
git clone https://github.com/XavierKdG/CVMatching.git
cd CVMatching
```

### 2. Add Dataset
download the datasets below and place the it in the `data/raw/` directory

- [Resume.csv](https://www.kaggle.com/datasets/014ce6313a60bf1563ff9ef3d57879bd8e7c1e1be0e8926bffb82d51ee85fda8?select=Resume.csv)

- [job_descriptions2.csv](https://www.kaggle.com/datasets/014ce6313a60bf1563ff9ef3d57879bd8e7c1e1be0e8926bffb82d51ee85fda8?select=job_descriptions2.csv)

### 3. Install Conda Environment

1. Install Miniconda:

```bash
curl -o miniconda.sh [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
bash miniconda.sh
source ~/.bashrc
rm miniconda.sh
```

2. Create the Conda environment:

```
conda env create -f environment.yml
```

3. Activate the environment:

```bash
conda activate cvmatching
```

### 4. Run Qdrant Database (via Docker)

This still uses Docker, but only for the database.

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Your Qdrant UI will be at `http://localhost:6334`.

### 5. Configure for Localhost

For this manual setup, you must change the config file.

1. Open `configs/config.yml`

2. Ensure the `url` points to `localhost`:

```yml
qdrant:
  url: "http://localhost:6333" # <-- Use localhost for this setup
```

### 6. Run the Data Pipeline
In your terminal (with the cvmatching env active), run the full pipeline to populate Qdrant:

```bash
python pipeline.py
```

### 7. Run the Streamlit App

Finally, launch the app:
```bash
streamlit run app.py
```

Your Streamlit app will be at `http://localhost:8501`.


## Advanced Usage & Configuration

All the main Python scripts (`pipeline.py`, `preprocess.py`, `train.py`, `upload_to_qdrant.py`) accept a `--config` argument to specify which configuration file to use.

By default, they all use `configs/config.yml`.

If you want to run the pipeline with a different configuration (e.g., `configs/config1.yml`), you can run the following command:

```bash
python pipeline.py --config configs/config1.yml
```

This also works for individual scripts:
```bash
python -m src.preprocess --config configs/config1.yml
python -m src.train --config configs/config1.yml
```

## Project Structure
```
.
├── app.py              # The Streamlit UI script
├── configs/            # All configuration files
│   ├── config.yml
│   ├── config1.yml
│   └── spacy_config.cfg
├── data/               # Project data
│   ├── processed/      # Cleaned/processed data (Parquet, .spacy)
│   └── raw/            # Raw input datasets (CSVs, etc.)
├── docs/               # Project documentation
├── logs/               # Log files
├── models/             # Trained models (e.g., model-best)
├── notebooks/          # Jupyter notebooks for experimentation
├── results/            # Evaluation results
├── src/                # All Python source code
│   ├── __init__.py
│   ├── evaluate.py     # Core logic for matching (used by app.py)
│   ├── preprocess.py   # Script for data cleaning and prep
│   ├── train.py        # Script for training custom NER model
│   ├── upload_to_qdrant.py # Script to create embeddings and upload
│   └── utils.py        # Helper functions (logging, config loading)
├── Dockerfile          # Defines the Streamlit app container
├── docker-compose.yml  # Manages all services (app, qdrant, doccano)
├── environment.yml     # Lists all Conda/Pip dependencies
├── pipeline.py         # Main script to run all data processes
└── README.md           # This file
```