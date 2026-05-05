# compare_models.py
# Lancer depuis le terminal : python compare_models.py
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import json, os
 
ml_client = MLClient.from_config(DefaultAzureCredential())
 
# ⚠️  Remplacer par vos noms de jobs
PIPELINE_JOB = 'frosty_fly_j383kvfps7'   # ex: quirky_angle_lpb3lv45w5
AUTOML_JOB   = 'dreamy_snail_c2d2nmnnvw'     # ex: magenta_ghost_ppsw4xzh8x
 
# --- Métriques du modèle manuel (pipeline) ---
os.makedirs('/tmp/metrics_manuel', exist_ok=True)
ml_client.jobs.download(
    name=PIPELINE_JOB,
    download_path='/tmp/metrics_manuel/',
    output_name='metrics'
)
with open('/tmp/metrics_manuel/named-outputs/metrics/metrics.json') as f:
    m = json.load(f)
 
# --- Métriques du meilleur modèle AutoML ---
child_runs = list(ml_client.jobs.list(parent_job_name=AUTOML_JOB))
completed = [r for r in child_runs
             if r.status == 'Completed'
             and r.properties.get('score') not in (None, 'N/A')]
best = min(completed,
           key=lambda r: float(r.properties['score']))
automl_nrmse = float(best.properties['score'])
 
# --- Affichage comparatif ---
print('=' * 50)
print('  COMPARAISON : Manuel vs AutoML')
print('=' * 50)
print()
print(f'Modèle manuel ({m["model"]}) :')
print(f'  MAE  : {m["mae"]:.2f}')
print(f'  RMSE : {m["rmse"]:.2f}')
print(f'  R²   : {m["r2"]:.4f}')
print(f'  NRMSE   : {m["nrmse"]:.4f}')
print()
print(f'Modèle AutoML (best child: {best.name}) :')
print(f'  NRMSE : {automl_nrmse:.4f}')

print()
print('→ Déployer le meilleur : modifier deploy_endpoint.py')
print('  model=azureml:diabetes-predictor:1  (manuel)')
print('  model=azureml:diabetes-predictor:2  (AutoML)')
 