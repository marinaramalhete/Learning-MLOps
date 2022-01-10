import mlflow
logged_model = 'file:///Users/marinaramalhetedesouza/Documents/MLflow/mlflow_example/mlflow/notebooks/mlruns/1/3eb7700937874ed2aed7e5b1bff9cbae/artifacts/xgboost'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

data = pd.read_csv('data/processed/casas_X.csv')
predicted = loaded_model.predict(data)

data['predicted'] = predicted
data.to_csv('precos.csv')