# src/train.py
from azure.ai.ml.entities import CommandComponent
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
import os
 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
train_diabetes = CommandComponent(
    name='train_diabetes',
    version='1',
    display_name='3 - Train Diabetes Model',
    environment='azureml:diabetes-ml-env:1',
    code=PROJECT_ROOT,
    command=(
        'python src/train_script.py '
        '--train_data ${{inputs.train_data}} '
        '--alpha ${{inputs.alpha}} '
        '--model_output ${{outputs.model_output}}'
    ),
    inputs={
        'train_data': Input(type=AssetTypes.URI_FOLDER),
        'alpha':      Input(type='number', default=1.0),
    },
    outputs={'model_output': Output(type=AssetTypes.URI_FOLDER)},
)