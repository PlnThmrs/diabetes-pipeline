# deploy_endpoint.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint, ManagedOnlineDeployment,
    Environment, CodeConfiguration)
from azure.identity import DefaultAzureCredential
import os
 
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ml_client = MLClient.from_config(DefaultAzureCredential())
 
# ⚠️  VARIABLE UNIQUE — utilisée partout pour éviter les incohérences
ENDPOINT_NAME = 'diabetes-endpoint-[prenom]'  # ex: diabetes-endpoint-untel
 
ep = ManagedOnlineEndpoint(name=ENDPOINT_NAME, auth_mode='key')
ml_client.online_endpoints.begin_create_or_update(ep).result()
print('✓ Endpoint créé')
 
deploy = ManagedOnlineDeployment(
    name='blue',
    endpoint_name=ENDPOINT_NAME,
    model='azureml:diabetes-predictor:1', #Modèle manuel
    code_configuration=CodeConfiguration(
        code=os.path.join(PROJECT_ROOT, 'src'),
        scoring_script='score.py'
    ),
    environment=Environment(
        name='diabetes-deploy-env',
        conda_file=f'{PROJECT_ROOT}/conda.yml',
        image='mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest'),
    instance_type='Standard_F2s_v2',
    instance_count=1,
)
ml_client.online_deployments.begin_create_or_update(deploy).result()
print('✓ Déploiement terminé')
 
ep.traffic = {'blue': 100}
ml_client.online_endpoints.begin_create_or_update(ep).result()
 
ep   = ml_client.online_endpoints.get(ENDPOINT_NAME)
keys = ml_client.online_endpoints.get_keys(ENDPOINT_NAME)
print(f'URL : {ep.scoring_uri}')
print(f'Clé : {keys.primary_key[:30]}...')

