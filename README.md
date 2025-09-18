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

Download de 2 datasets en voeg ze toe aan de folder `data/raw/`

### 3. Miniconda omgeving installeren
1. Installeer Miniconda (en verwijder installatie bestand):

`curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`

`bash miniconda.sh`

`source ~/.bashrc`

`rm miniconda.sh` 

2. Maak de Conda environment aan:

`conda env create -f environment.yml`

3. Activeer de environment:

`conda activate cvmatching`

4. Installeer PyTorch (CPU-versie)

`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

### 4. Preprocessing uitvoeren

`python src/preprocess.py --input data/raw --output data/processed`

### 5. Model trainen

`python src/train.py --config configs/train_config.yml`

### 6. Model evalueren

`python src/evaluate.py --model_path models/yourmodel.pth --data_path data/processed`
