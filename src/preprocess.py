# src/preprocess.py
from azure.ai.ml.entities import CommandComponent
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
import os
 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
preprocess_diabetes = CommandComponent(
    name='preprocess_diabetes',
    version='1',
    display_name='1 - Preprocess Diabetes',
    environment='azureml:diabetes-ml-env:1',
    code=PROJECT_ROOT,
    command=(
        'python src/preprocess_script.py '
        '--raw_data ${{inputs.raw_data}} '
        '--output_data ${{outputs.output_data}}'
    ),
    inputs={'raw_data': Input(type=AssetTypes.URI_FILE)},
    outputs={'output_data': Output(type=AssetTypes.URI_FOLDER)},
)