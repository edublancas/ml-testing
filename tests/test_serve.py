from pathlib import Path

import yaml
import pytest
from ploomber.spec import DAGSpec
from ploomber.exceptions import DAGBuildError

root_path = Path(__file__).parent.parent


def test_error_on_incorrect_number_of_input_columns(tmp_path):
    # switch script that loads data for one that gets corruped data
    spec = yaml.safe_load(Path('pipeline.serve.yaml').read_text())
    spec['tasks'][0]['source'] = 'tasks/get-fake.py'

    # store output in a temporary directory to avoid overwriting our results
    env = dict(products_root=tmp_path, here=root_path)

    # load pipeline
    dag = DAGSpec(spec, env=env).to_dag().render()

    # execute pipeline, it should raise an exception
    with pytest.raises(DAGBuildError) as excinfo:
        dag.build()

    # ensure that the right error message is displayed
    assert 'Wrong number of columns' in str(excinfo.value)
