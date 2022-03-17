import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE

def regression_report(y_true, pred, verbose = True):
    mse = MSE(y_true, pred)
    mae = MAE(y_true, pred)
    rmse = np.sqrt(mse)
    mape = MAPE(y_true, pred)
    
    if verbose:
        print(f'mse = {mse:.4f}')
        print(f'mae = {mae:.4f}')
        print(f'rmse = {rmse:.4f}')
        print(f'mape = {mape:.4f}')

    # return mse, mae, rmse, mape