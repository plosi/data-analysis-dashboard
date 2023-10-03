import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb

def run_arima(train, test, target, order, seasonal_order, exog_feat):
    if exog_feat is not None:
        # exog_ = train[[c for c in exog_feat]],
        exog_ = train[['sst', 'max_temp', 'rainfall', 'enso_anom']]
    else:
        exog_ = None

    model = ARIMA(train[target],
            exog=exog_,
            order=order,
            seasonal_order=seasonal_order,
            ).fit()
    
    start = len(train)
    end = len(train) + len(test) - 1

    if exog_feat is not None:
        # pred = model.predict(start, end, exog=test[[c for c in exog_feat]]).rename('Prediction')
        pred = model.predict(start, end, exog=test[['sst', 'max_temp', 'rainfall', 'enso_anom']]).rename('Prediction')
    else:
        pred = model.predict(start, end).rename('Prediction')

    rmse = np.sqrt(mean_squared_error(test[target], pred))
    mae = mean_absolute_error(test[target], pred)

    res = {
        'order': order,
        'seasonal_order': seasonal_order,
        'exog_feat': exog_feat,
        'train': train[target],
        'test': test[target],
        'prediction': pred,
        'rmse': rmse,
        'mae': mae,
        'mean': test[target].mean(),
        'std': test[target].std(),
        'train_size': len(train),
        'test_size': len(test),
        'columns': (train.columns, test.columns)
    }

    return res


def run_gboost_model(model_name, X_train, X_test, y_train, y_test, params):

    if model_name == 'XGBoost Regressor':

        model = xgb.XGBRegressor(
            subsample=0.5,
            learning_rate = params['learning_rate'],
            n_estimators = params['n_estimators'],
            max_depth = params['max_depth'],
            early_stopping_rounds = params['early_stop'],
        )

        model.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)

        y_pred = pd.Series(model.predict(X_test), index=y_test.index)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        feature_importance = pd.DataFrame(data=model.feature_importances_,
                                        index=model.feature_names_in_,
                                        columns=['importance']).sort_values('importance', ascending=False)

        res = {
            'params': params,
            'features': X_train.columns,
            'train': y_train,
            'test': y_test,
            'feature_importance': feature_importance,
            'prediction': y_pred,
            'rmse': rmse,
            'mae': mae,
            'mean': y_test.mean(),
            'std': y_test.std(),
            'train_size': len(X_train),
            'test_size': len(X_test),
        }
    else:
        print('Error')
        res = {'features':'Something went wrong'}

    return res



def plot_model_results(train, test, pred):
    fig, ax = plt.subplots()

    ax.plot(train)#, legend=True, color='C1')
    ax.plot(test)#, legend=True, color='C2')
    ax.plot(pred)#, legend=True, color='C3')

    # fig = go.Figure()
    
    # fig.add_trace(go.Scatter(train))

    # fig.add_trace(go.Scatter(test))
    # fig.add_trace(go.Scatter(pred))

    return fig