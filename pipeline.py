# pipeline.py
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes
 
from src.preprocess          import preprocess_diabetes
from src.feature_engineering import feature_engineering_diabetes
from src.train               import train_diabetes
from src.evaluate            import evaluate_diabetes
 
 
@pipeline(
    name='diabetes_pipeline',
    description='Pipeline Diabetes : preprocess → features → train → evaluate',
    compute='cluster-cpu',
)
def diabetes_pipeline(raw_data, alpha=1.0):
    step_pre = preprocess_diabetes(
        raw_data=raw_data,
    )
    step_fe = feature_engineering_diabetes(
        input_data=step_pre.outputs.output_data,
    )
    step_train = train_diabetes(
        train_data=step_fe.outputs.train_output,
        alpha=alpha,
    )
    step_eval = evaluate_diabetes(
        model_input=step_train.outputs.model_output,
        test_data=step_fe.outputs.test_output,
    )
    return {        
        # Outputs exposés pour AutoML (Étape 18) — contiennent train_data.csv / test_data.csv
        'pipeline_train_data': step_fe.outputs.train_output,
        'pipeline_test_data': step_fe.outputs.test_output,
        'model':   step_train.outputs.model_output,
        'metrics': step_eval.outputs.eval_output,
    }