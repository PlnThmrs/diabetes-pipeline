# src/feature_engineering_script.py
import argparse, os, logging, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
 
parser = argparse.ArgumentParser()
parser.add_argument('--input_data',   type=str,   required=True)
parser.add_argument('--train_output', type=str,   required=True)
parser.add_argument('--test_output',  type=str,   required=True)
parser.add_argument('--test_size',    type=float, default=0.2)
parser.add_argument('--random_state', type=int,   default=42)
args = parser.parse_args()
 
X = pd.read_csv(f'{args.input_data}/X.csv')
y = pd.read_csv(f'{args.input_data}/y.csv').squeeze()
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state)
log.info(f'Train: {X_train.shape} | Test: {X_test.shape}')
 
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)
 
os.makedirs(args.train_output, exist_ok=True)
X_train_s.to_csv(f'{args.train_output}/X_train.csv', index=False)
y_train.to_csv(  f'{args.train_output}/y_train.csv', index=False)
joblib.dump(scaler, f'{args.train_output}/scaler.pkl')

# --- Export CSV complet (X+y) pour AutoML (Étape 18) ---
train_full = pd.concat([X_train_s, y_train.reset_index(drop=True)], axis=1)
train_full.to_csv(f'{args.train_output}/train_data.csv', index=False)
 
os.makedirs(args.test_output, exist_ok=True)
X_test_s.to_csv(f'{args.test_output}/X_test.csv', index=False)
y_test.to_csv(  f'{args.test_output}/y_test.csv', index=False)

# Export test complet (X+y) pour AutoML
test_full = pd.concat([X_test_s, y_test.reset_index(drop=True)], axis=1)
test_full.to_csv(f'{args.test_output}/test_data.csv', index=False)

log.info('Feature engineering terminé.')
log.info('Fichiers de test (séparés et fusionnés) générés avec succès.')