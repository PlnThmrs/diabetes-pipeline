# src/evaluate.py
from azure.ai.ml.entities import CommandComponent
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
import os
 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
evaluate_diabetes = CommandComponent(
    name='evaluate_diabetes',
    version='1',
    display_name='4 - Evaluate Diabetes Model',
    environment='azureml:diabetes-ml-env:1',
    code=PROJECT_ROOT,
    command=(
        'python src/evaluate_script.py '
        '--model_input ${{inputs.model_input}} '
        '--test_data ${{inputs.test_data}} '
        '--eval_output ${{outputs.eval_output}}'
    ),
    inputs={
        'model_input': Input(type=AssetTypes.URI_FOLDER),
        'test_data':   Input(type=AssetTypes.URI_FOLDER),
    },
    outputs={'eval_output': Output(type=AssetTypes.URI_FOLDER)},
)