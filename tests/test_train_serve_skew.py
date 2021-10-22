from pathlib import Path

import yaml
from ploomber.spec import DAGSpec
import pandas as pd

root_path = Path(__file__).parent.parent


def test_trainig_serving_skew(tmp_path):
    # store output in a temporary directory to avoid overwriting our results
    env = dict(products_root=tmp_path, here=root_path)

    # load training data but replace first task for the one with fake data
    spec_training = yaml.safe_load(Path('pipeline.yaml').read_text())
    spec_training['tasks'][0]['source'] = 'tasks/get-fake.py'
    # ensure it generates the fake target variable
    spec_training['tasks'][0]['params'] = dict(type_='fake_training')
    # we don't want to check data quality here
    del spec_training['tasks'][2]['on_finish']

    # load serving data but replace first task for the one with fake data
    spec_serving = yaml.safe_load(Path('pipeline.serve.yaml').read_text())
    spec_serving['tasks'][0]['source'] = 'tasks/get-fake.py'
    # ensure it doesn't add any extra columns
    spec_serving['tasks'][0]['params'] = dict(type_=None)

    # load and build training pipeline (up until the clean task)
    dag_train = DAGSpec(spec_training, env=env).to_dag().render()
    dag_train.build_partially('clean')

    # load and build serving pipeline (up until the clean task)
    dag_serve = DAGSpec(spec_serving, env=env).to_dag().render()
    dag_serve.build_partially('clean')

    # load generated features by the training and serving pipelines
    features_train = pd.read_csv(dag_train['clean'].product['data'])
    features_train = features_train.drop('target', axis='columns')
    features_serve = pd.read_csv(dag_serve['clean'].product['data'])

    # both should be the same
    assert features_train.equals(features_serve)