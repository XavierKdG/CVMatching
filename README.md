# CVMatching
Project Semester 5 - Groep **Weekend**

## Over het project
CVMatching is een project waarin wij met behulp van NLP, cv's automatisch analyseren en matchen met vacatures.

---

## Installatie & Gebruik

### 1. Repository clonen
`git clone https://github.com/XavierKdG/CVMatching.git`

`cd CVMatching`

### 2. Datasets toevoegen

Download de 2 datasets en voeg ze toe aan de folder 'data'

### 3. Miniconda installeren
1. `curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`

2. `bash miniconda.sh`

3. `source ~/.bashrc`

### 4. Miniconda.sh verwijderen
Verwijder de `miniconda.sh` file

### 5. Conda environment aanmaken
`conda env create -f environment.yml`

### 6. Environment activeren

`conda activate cvmatching`

### 7. Installeer PyTorch

`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

