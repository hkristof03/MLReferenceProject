from pathlib import Path
import requests
from uuid import uuid4

import pandas as pd


n = 1000
base_url = 'http://localhost:8000'
#base_url = 'http://10.106.231.9:8000'
endpoint = '/ml_project_reference/inference'

path_test_data = Path().resolve().joinpath(
    "train", "data", "test.parquet"
)
df_test = pd.read_parquet(path_test_data)
df_test = df_test.sample(n=n, replace=False)

for record in df_test[['narrative', 'product']].to_dict(orient='records'):

    data = {**record, 'user_id': str(uuid4())}

    response = requests.post(
        base_url + endpoint,
        json=data
    )
