import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from statsmodels.tsa.seasonal import seasonal_decompose

def make_lineplot(x, y, title=''):
    fig = px.line(x=x,
                  y=y,
                  title=title)
    return fig

def make_scatterplot(x, y, title=''):
    fig = px.scatter(x=x,
                     y=y,
                     title=title)
    return fig

def make_barplot(x, y, title=''):
    fig = px.bar(x=x,
                 y=y,
                 title=title)
    return fig

def make_hist(df, x, title=''):
    fig = px.histogram(df,
                       x=x,
                       title=title)
    return fig

def make_boxplot(df, y, title=''):
    fig = px.box(df,
                 y=y,
                 title=title)
    return fig


# ----- MODEL PLOTS ----
def make_decomposition_plots(series, model='additive'):
    trend = seasonal_decompose(x=series, model=model).trend
    seasonal = seasonal_decompose(x=series, model=model).seasonal
    residual = seasonal_decompose(x=series, model=model).resid

    tmp = [trend, seasonal, residual]
    decompose_df = pd.DataFrame(tmp).T
    decompose_df.columns = ['trend', 'seasonal', 'resid']

    fig = make_subplots(rows=4, cols=1)
    
    fig.append_trace(
        go.Scatter(
            x=decompose_df.index,
            y=series,
            name=''
            ),
            row=1,
            col=1
    )

    fig.append_trace(
        go.Scatter(
            x=decompose_df.index,
            y=decompose_df.trend,
            name='Trend'
            ),
            row=2,
            col=1
    )

    fig.append_trace(
        go.Scatter(
            x=decompose_df.index,
            y=decompose_df.seasonal,
            name='Seasonal'
            ),
            row=3,
            col=1
    )

    fig.append_trace(
        go.Scatter(
            x=decompose_df.index,
            y=decompose_df.resid,
            mode='markers',
            name='Residual'
            ),
            row=4,
            col=1
    )

    # fig = seasonal_decompose(x=x, model=model).plot()
    fig.update_layout(height=800, width=800, title_text="Decomposition Plot")
    return fig