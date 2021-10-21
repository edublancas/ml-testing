import shutil
from invoke import task
from pathlib import Path
from ploomber.spec import DAGSpec


@task
def store_ref_data(c):
    """Store reference data
    """
    dag = DAGSpec('pipeline.yaml').to_dag()
    # ensure we have up-to-date results
    dag.build()

    # copy clean data to the reference folder
    # NOTE: in a real project, we shouldn't save data in the repository
    # but in a storage system like S3 or similar
    ref = Path('reference')
    ref.mkdir(exist_ok=True)
    shutil.copy(dag['clean'].product['data'], ref / 'clean.csv')