$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

New-Item -ItemType Directory -Force .\checkpoints\logs | Out-Null

& "c:/Users/LENOVO/Desktop/4th Sem/Artificial Intelligence/AI Project/Drug Interaction Predictor/Files/ddi-twosides-final/ddi-twosides/.venv/Scripts/python.exe" train.py --data data/TWOSIDES.csv.gz --max_pairs 20000 --epochs 1 2>&1 | Tee-Object -FilePath .\checkpoints\logs\train_cache.txt
& "c:/Users/LENOVO/Desktop/4th Sem/Artificial Intelligence/AI Project/Drug Interaction Predictor/Files/ddi-twosides-final/ddi-twosides/.venv/Scripts/python.exe" train.py --data data/TWOSIDES.csv.gz --max_pairs 20000 --epochs 60 2>&1 | Tee-Object -FilePath .\checkpoints\logs\train_main.txt
& "c:/Users/LENOVO/Desktop/4th Sem/Artificial Intelligence/AI Project/Drug Interaction Predictor/Files/ddi-twosides-final/ddi-twosides/.venv/Scripts/python.exe" train_rl.py --data data/TWOSIDES.csv.gz --max_pairs 20000 --episodes 100 2>&1 | Tee-Object -FilePath .\checkpoints\logs\train_rl.txt
