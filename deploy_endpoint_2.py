# deploy_endpoint_automl_2.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment 
from azure.identity import DefaultAzureCredential
import os 

ml_client = MLClient.from_config(DefaultAzureCredential()) 

# À adapter
ENDPOINT_NAME = 'diabetes-endpoint-[prenom]' 
MODEL_NAME    = 'azureml:diabetes-predictor-automl:2' # Le nom de votre modèle AutoML

# 1. Création de l'Endpoint
ep = ManagedOnlineEndpoint(name=ENDPOINT_NAME, auth_mode='key') 
ml_client.online_endpoints.begin_create_or_update(ep).result() 
print('✓ Endpoint créé') 

# 2. Déploiement (Version simplifiée pour AutoML/MLflow)
deploy = ManagedOnlineDeployment(     
    name='blue',     
    endpoint_name=ENDPOINT_NAME,     
    model=MODEL_NAME,     
    # Pas besoin de code_configuration ni d'environment !    
    instance_type='Standard_F2s_v2',     
    instance_count=1, ) 
print('Déploiement du modèle AutoML en cours...') 
ml_client.online_deployments.begin_create_or_update(deploy).result() 

# 3. Gestion du trafic
ep.traffic = {'blue': 100} 
ml_client.online_endpoints.begin_create_or_update(ep).result() 

# 4. Récupération des infos
ep = ml_client.online_endpoints.get(ENDPOINT_NAME) 
keys = ml_client.online_endpoints.get_keys(ENDPOINT_NAME) 
print(f'✓ Déploiement terminé') 
print(f'URL : {ep.scoring_uri}') 
print(f'Clé : {keys.primary_key[:30]}...')