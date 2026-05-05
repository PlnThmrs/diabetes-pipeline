# src/feature_engineering.py
from azure.ai.ml.entities import CommandComponent
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
import os
 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
feature_engineering_diabetes = CommandComponent(
    name='feature_engineering_diabetes',
    version='1',
    display_name='2 - Feature Engineering Diabetes',
    environment='azureml:diabetes-ml-env:1',
    code=PROJECT_ROOT,
    command=(
        'python src/feature_engineering_script.py '
        '--input_data ${{inputs.input_data}} '
        '--train_output ${{outputs.train_output}} '
        '--test_output ${{outputs.test_output}}'
    ),
    inputs={'input_data': Input(type=AssetTypes.URI_FOLDER)},
    outputs={
        'train_output': Output(type=AssetTypes.URI_FOLDER),
        'test_output':  Output(type=AssetTypes.URI_FOLDER),
    },
)
