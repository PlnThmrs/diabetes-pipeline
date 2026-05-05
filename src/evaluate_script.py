# src/evaluate_script.py
import argparse, os, json, logging, joblib
import pandas as pd, numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
 
parser = argparse.ArgumentParser()
parser.add_argument('--model_input', type=str, required=True)
parser.add_argument('--test_data',   type=str, required=True)
parser.add_argument('--eval_output', type=str, required=True)
args = parser.parse_args()
 
model  = joblib.load(f'{args.model_input}/model.pkl')
X_test = pd.read_csv(f'{args.test_data}/X_test.csv')
y_test = pd.read_csv(f'{args.test_data}/y_test.csv').squeeze()
 
y_pred = np.maximum(model.predict(X_test), 0)
 
mae  = float(mean_absolute_error(y_test, y_pred))
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2   = float(r2_score(y_test, y_pred))
metrics_num = {'mae': mae, 'rmse': rmse, 'r2': r2, 'n_test': int(len(y_test))}
log.info(f'MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.4f}')
 
os.makedirs(args.eval_output, exist_ok=True)
 
# MLflow — pas de start_run() ici
mlflow.log_param('model_name', type(model).__name__)
mlflow.log_metrics(metrics_num)
 
# Graphique
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_test, y_pred, alpha=0.4, s=20, color='steelblue')
lim = max(y_test.max(), y_pred.max())
axes[0].plot([0, lim], [0, lim], 'r--', lw=2)
axes[0].set(xlabel='Réel', ylabel='Prédit', title=f'MAE={mae:.2f}')
axes[1].hist(y_test.values - y_pred, bins=30, color='steelblue', alpha=0.7)
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set(xlabel='Résidus', title=f'RMSE={rmse:.2f}')
plt.tight_layout()
fig.savefig(f'{args.eval_output}/diagnostics.png', dpi=150)
mlflow.log_artifact(f'{args.eval_output}/diagnostics.png')
plt.close()
 
# JSON
metrics_full = {**metrics_num, 'model': type(model).__name__}
json_path = f'{args.eval_output}/metrics.json'
with open(json_path, 'w') as fh:
    json.dump(metrics_full, fh, indent=2)
mlflow.log_artifact(json_path)
log.info('Évaluation terminée.')