# src/score.py
import json, os, logging, joblib
import pandas as pd, numpy as np
 
model = None
scaler = None
FEATURE_NAMES = ['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']
 
def init():
    global model, scaler
    # 1. Récupérer le chemin de base (inclut déjà 'diabetes-predictor/1')
    model_root = os.getenv('AZUREML_MODEL_DIR')
 
    # 2. Pointer vers model_output
    model_dir = os.path.join(model_root, 'model_output')
 
    model_path = os.path.join(model_dir, 'model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
 
    # Debug pour vérifier dans les logs
    print(f'Tentative de chargement depuis : {model_path}')
 
    if not os.path.exists(model_path):
        content = os.listdir(model_root)
        raise FileNotFoundError(f'Fichier absent. Contenu de {model_root} : {content}')
 
    model = joblib.load(model_path)
 
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
 
    logging.info('✓ Modèle et Scaler chargés avec succès.')
 
def run(raw_data):
    try:
        payload = json.loads(raw_data)
        records = payload.get('data', payload)
        df = pd.DataFrame(records)
 
        preds = model.predict(df)
 
        # --- Output Documenté (dict direct, PAS json.dumps) ---
        return {
            "model_info": {
                "name": "Diabetes Progression Predictor",
                "version": os.getenv("AZUREML_MODEL_VERSION", "1"),
                "type": "Regression"
            },
            "results": [
                {
                    "input_index": i,
                    "prediction": round(float(p), 2),
                    "unit": "quantitative score",
                    "status": "success"
                } for i, p in enumerate(preds)
            ],
            "metadata": {
                "total_records": len(preds),
                "feature_names": FEATURE_NAMES
            }
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}
