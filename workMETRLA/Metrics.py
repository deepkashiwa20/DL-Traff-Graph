import numpy as np

def evaluate(y_true, y_pred, precision=10):
    # print('MSE:', round(MSE(y_true, y_pred), precision))
    # print('RMSE:', round(RMSE(y_true, y_pred), precision))
    # print('MAE:', round(MAE(y_true, y_pred), precision))
    # print('MAPE:', round(MAPE(y_true, y_pred), precision), '%')
    # print('PCC:', round(PCC(y_true, y_pred), precision))
    return MSE(y_true, y_pred), RMSE(y_true, y_pred), MAE(y_true, y_pred), MAPE(y_true, y_pred)

def MSE(y_true, y_pred):
    y_true[y_true < 1e-5] = 0
    y_pred[y_pred < 1e-5] = 0
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mse = np.square(y_pred - y_true)
        mse = np.nan_to_num(mse * mask)
        mse = np.mean(mse)
        return mse
    
def RMSE(y_true, y_pred):
    y_true[y_true < 1e-5] = 0
    y_pred[y_pred < 1e-5] = 0
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse
        
def MAE(y_true, y_pred):
    y_true[y_true < 1e-5] = 0
    y_pred[y_pred < 1e-5] = 0
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae

def MAPE(y_true, y_pred, null_val=0):
    y_true[y_true < 1e-5] = 0
    y_pred[y_pred < 1e-5] = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100
    
# def MSE(y_true, y_pred):
#     return np.mean(np.square(y_pred - y_true))

# def RMSE(y_true, y_pred):
#     return np.sqrt(MSE(y_pred, y_true))

# def MAE(y_true, y_pred):
#     return np.mean(np.abs(y_pred - y_true))

# def MAPE(y_pred:np.array, y_true:np.array, epsilon=1e-3):       # avoid zero division
#     return np.mean(np.abs(y_pred - y_true) / np.clip((np.abs(y_pred) + np.abs(y_true)) * 0.5, epsilon, None))
    
# def PCC(y_pred:np.array, y_true:np.array):      # Pearson Correlation Coefficient
#     return np.corrcoef(y_pred.flatten(), y_true.flatten())[0,1]




