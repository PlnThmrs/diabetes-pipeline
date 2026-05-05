# diabetes-pipeline
Construction d'un pipeline de machine learning type régression sur Azure ML

# Etapes à réaliser
1. Créer dans Azure :
Workspace         : ws-[prénom]         → Région : votre région choisie
Compute Instance  : instance-[prénom]   → Même région
Compute Cluster   : cluster-cpu         → Même région

2. Une fois la compute instance créée, ouvrir Jupyter Labs pour générer le token (on peut fermer ensuite)

3. Lancer ensuite VS code desktop

4. Créer le dossier : 
```
cd ~/cloudfiles/code
mkdir -p diabetes-pipeline/
```
5. Importer les fichiers dans azure : Workspace > Notebooks > copier/coller dans diabetes-pipeline
Rq: Dans le fichier conda.yml, les versions sont épinglées pour garantir la compatibilité sur North Europe et West Europe.
mlflow et mlflow-skinny doivent avoir la même version (2.15.1). Un mismatch provoque des erreurs d'API Azure ML. marshmallow<3.20.0 est requis pour azure-ai-ml==1.12.0.

6. Créer l'environnement conda et le kernel Jupyter
```
cd ~/cloudfiles/code/diabetes-pipeline
 
# Créer l'environnement (5 à 10 minutes)
# ⚠️  conda va demander d'accepter les Terms of Service
#     pour les channels "main" et "r" → taper "a" pour accepter
conda env create -f conda.yml
 
# Activer
conda activate diabetes-ml-env
 
# Vérifications
python -c "import azure.ai.ml; print(azure.ai.ml.__version__)"
# → 1.12.0
 
python -c "import mlflow; print(mlflow.__version__)"
# → 2.15.1
 
python -c "from azure.ai.ml.entities import CommandComponent; print('OK')"
# → OK
 
# Installer le kernel Jupyter
python -m ipykernel install --user --name diabetes-ml-env \
    --display-name "Python (diabetes-ml-env)"
```
Rq: Si pkg_resources est manquant : pip install setuptools==70.0.0 --force-reinstall

7. Configurer la connexion Azure

```
# ⚠️  AVANT de tester — Initialiser le SSO de la Compute Instance :
#     Studio → Compute → cliquer sur votre instance → cliquer "JupyterLab"
#     Attendre que JupyterLab s'ouvre, puis revenir ici
#     (nécessaire une seule fois après chaque redémarrage de l'instance)
 
# Se connecter à Azure (ouvre le navigateur)
az login

# Récupérer votre subscription_id
az account show --query id -o tsv
 
# ⚠️  IMPORTANT — Remplacer les 3 valeurs ci-dessous par VOS informations :
#   - VOTRE_SUBSCRIPTION_ID : le résultat de az account show ci-dessus
#   - rg-mlops-[prénom]     : votre resource group (ex: rg-mlops-rahim)
#   - ws-[prénom]           : votre workspace (ex: ws-rahim)
# Créer config.json
cat > ~/cloudfiles/code/diabetes-pipeline/config.json << 'EOF'
{
    "subscription_id": "VOTRE_SUBSCRIPTION_ID",
    "resource_group":  "rg-mlops-[prénom]",
    "workspace_name":  "ws-[prénom]"
}
EOF
 
# ⚠️  Activer l'environnement conda AVANT de tester
conda activate diabetes-ml-env

# Tester
cd ~/cloudfiles/code/diabetes-pipeline
python -c "
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
ml = MLClient.from_config(DefaultAzureCredential())
print(f'Workspace : {ml.workspace_name}')"
```
La commande doit afficher votre vrai workspace : Workspace : ws-untel (pas ws-[prénom])

8. Créer le dataset Diabetes
```
cd ~/cloudfiles/code/diabetes-pipeline
 
python << 'PYEOF'
import pandas as pd
from sklearn.datasets import load_diabetes
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import os
 
# Générer le CSV
data = load_diabetes(as_frame=True)
df = data.frame
os.makedirs('/tmp', exist_ok=True)
df.to_csv('/tmp/diabetes.csv', index=False)
print(f'CSV : {df.shape}')
 
# Uploader comme Data Asset
ml_client = MLClient.from_config(DefaultAzureCredential())
asset = ml_client.data.create_or_update(Data(
    path='/tmp/diabetes.csv',
    type=AssetTypes.URI_FILE,
    name='diabetes-dataset',
    version='1',
    description='Sklearn diabetes — 442 lignes, 10 features + target'
))
print(f'Data Asset : {asset.name} v{asset.version}')
PYEOF
```
Pour vérifier : Studio → Data → Data assets : diabetes-dataset v1 visible.

9. Créer l'environnement Azure ML
```
cd ~/cloudfiles/code/diabetes-pipeline
 
python << 'PYEOF'
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
 
ml_client = MLClient.from_config(DefaultAzureCredential())
env = Environment(
    name='diabetes-ml-env',
    version='1',
    conda_file='./conda.yml',
    image='mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest',
)
env = ml_client.environments.create_or_update(env)
print(f'Environment : {env.name} v{env.version}')
print('Build en cours (5-10 min) — attendre Succeeded dans Studio → Environments')
PYEOF
```
Pour vérifier : Studio → Environments → diabetes-ml-env → Version 1 → onglet « Build log ». Statut attendu : Succeeded. Si le build est encore en cours (Building), attendre 5-10 min. Si pas d'onglet Build log visible, le build se lancera automatiquement au premier job — vous pouvez continuer les étapes suivantes en attendant.

10. Lancer le notebook pipeline_notebook pour lancer le premier job

11. Vérfier les outputs du pipeline dans Studio
Vérifier l'existence de model.pkl :
- Studio → Jobs → cliquer sur le pipeline job Completed
- Dans le graphe du pipeline, cliquer sur le composant step_train
- Onglet Outputs + logs → dossier model/ → vérifier que model.pkl est présent
- Dans le même job, cliquer sur le composant step_eval
- Onglet Métriques → les valeurs MAE, RMSE, R² et n_test s'affichent directement
- Valeurs attendues pour le dataset Diabetes : MAE ≈ 42-43  |  RMSE ≈ 53-54  |  R² ≈ 0.45  |  n_test = 89
Si model.pkl est absent → le composant train a échoué. 
Si R² < 0.3 → vérifier les features (feature_engineering_script.py). Ne pas enregistrer un modèle dont les métriques sont anormales.

12. Enregistrer le modèle entraîné dans le Registry Azure ML pour pouvoir le déployer ensuite.
Avant de lancer ce script, il faut récupérer le nom exact du pipeline job :
- Studio → Jobs → cliquer sur le job dont le statut est Completed
- Copier le Nom du travail parent (ex : ashy_pummelo_vb391tsvrs) — PAS le Nom d'affichage
- Coller ce nom dans la variable PIPELINE_JOB du script register_model.py

13. Lancer submit_automl.py - Penser à modifier PIPELINE_JOB

14. Une fois tous les child job du autoML terminés (Studio → Jobs → diabetes-progression → votre job AutoML → Modèles + travaux enfants), récupérer le nom du job Parent AutoML et lancer register_automl.py

15. On comparer les 2 modèles avec compare_model.py mais vu que les métriques sont pas les mêmes ce n'est pas très utile, il vaut mieux passer par Azure ML studio

16. Lancer deploy_endpoint.py pour le premier modèle, deploy_endpoint_2 pour le modèle AutoML
Penser à changer le nom du endpoint

17. Lorsque l'endpoint est affiché comme "succeeded" dans Studio, lancer test_endpoint.py
Résultat attendu: Status : 200 | Prédictions : [154.8, 88.3]   (les valeurs exactes varient)
