# src/preprocess_script.py
import argparse, os, logging
import pandas as pd
 
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
 
parser = argparse.ArgumentParser()
parser.add_argument('--raw_data',    type=str, required=True)
parser.add_argument('--output_data', type=str, required=True)
args = parser.parse_args()
 
df = pd.read_csv(args.raw_data)
log.info(f'Dataset : {df.shape}')
 
X = df.drop(columns=['target'])
y = df[['target']]
log.info(f'Features : {X.shape} | Target : {y.shape}')
 
os.makedirs(args.output_data, exist_ok=True)
X.to_csv(f'{args.output_data}/X.csv', index=False)
y.to_csv(f'{args.output_data}/y.csv', index=False)
log.info('Preprocess terminé.')