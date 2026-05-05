# src/train_script.py
import argparse, os, logging, joblib, shutil
import pandas as pd
import mlflow
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
 
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
 
parser = argparse.ArgumentParser()
parser.add_argument('--train_data',   type=str,   required=True)
parser.add_argument('--alpha',        type=float, default=1.0)
parser.add_argument('--model_output', type=str,   required=True)
args = parser.parse_args()
 
X_train = pd.read_csv(f'{args.train_data}/X_train.csv')
y_train = pd.read_csv(f'{args.train_data}/y_train.csv').squeeze()
 
with mlflow.start_run():
    mlflow.log_param('alpha', args.alpha)
    mlflow.log_param('n_train', len(X_train))
 
    model = Ridge(alpha=args.alpha)
    model.fit(X_train, y_train)
 
    train_mae = float(mean_absolute_error(y_train, model.predict(X_train)))
    train_r2  = float(model.score(X_train, y_train))
    mlflow.log_metrics({'train_mae': train_mae, 'train_r2': train_r2})
    log.info(f'Train MAE={train_mae:.2f} | R2={train_r2:.4f}')
 
    os.makedirs(args.model_output, exist_ok=True)
    joblib.dump(model, f'{args.model_output}/model.pkl')
    mlflow.log_artifact(f'{args.model_output}/model.pkl', artifact_path='model')
 
    scaler_src = f'{args.train_data}/scaler.pkl'
    if os.path.exists(scaler_src):
        shutil.copy(scaler_src, f'{args.model_output}/scaler.pkl')
 
log.info('Entraînement terminé.')