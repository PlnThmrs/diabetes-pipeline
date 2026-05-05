# register_automl.py
# Lancer depuis le terminal : python register_automl.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import os, shutil
 
ml_client = MLClient.from_config(DefaultAzureCredential())
 
# Remplacer par le nom du job AutoML terminé
AUTOML_JOB = 'dreamy_snail_c2d2nmnnvw'  # Studio → Jobs → nom du job AutoML

# --- Étape 1 : Identifier le meilleur child job ---
# Récupérer le meilleur child job (meilleur score)
parent_job = ml_client.jobs.get(AUTOML_JOB)
best_child_run_id = parent_job.tags.get('model_explain_best_run_child_id')
best = ml_client.jobs.get(best_child_run_id)
algo = best.properties.get('run_algorithm', 'AutoML')
score = float(best.properties.get('score', 0))
print(f'Meilleur algorithme : {algo} — NRMSE={score:.4f}')

# --- Étape 2 : Télécharger les artifacts (inclut le modèle MLflow) ---
# Le path azureml://jobs/.../outputs/mlflow-model/ ne fonctionne pas
# car AutoML génère des noms d'output dynamiques.
# Méthode fiable : download local → trouver le dossier MLflow → register.
local = './automl_model_download'
if os.path.exists(local):
    shutil.rmtree(local)
ml_client.jobs.download(name=AUTOML_JOB, download_path=local, all=True)
 
# Trouver automatiquement le dossier MLflow dans named-outputs
named = f'{local}/named-outputs'
mlflow_dir = None
for d in os.listdir(named):
    if os.path.isfile(os.path.join(named, d, 'MLmodel')):
        mlflow_dir = os.path.join(named, d)
        break
 
if not mlflow_dir:
    print('ERREUR : dossier MLflow introuvable dans', named)
    exit(1)
print(f'Dossier MLflow trouvé : {mlflow_dir}')
 
# --- Étape 3 : Enregistrer le modèle ---
model_v2 = ml_client.models.create_or_update(Model(
    path=mlflow_dir,
    name='diabetes-predictor',
    version='2',
    description=f'AutoML best — NRMSE={score:.4f}',
    tags={'source': 'automl', 'nrmse': str(round(score, 4))},
    type=AssetTypes.MLFLOW_MODEL,
))
print(f'Enregistré : {model_v2.name} v{model_v2.version}')
print('Studio → Models → diabetes-predictor → comparer v1 (manuel) et v2 (AutoML)')
