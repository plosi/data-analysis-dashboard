# General imports
import pandas as pd
from sklearn.model_selection import train_test_split
import asyncio

# Shiny imports
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from shiny.types import FileInfo
import shinyswatch
from shinywidgets import render_widget

# Local imports
from data_page import layout as data_layout
from model_page import arima_layout, gradient_boost_layout
from plots import make_lineplot, make_scatterplot, make_barplot, make_boxplot, make_hist, make_decomposition_plots
from time_series import run_arima, plot_model_results, run_gboost_model

## App UI
app_ui = ui.page_fluid(
    shinyswatch.theme.minty(),
    
    ui.page_navbar(
        ui.nav('Data',
               data_layout),
        ui.nav_menu(
            'Model',
            'Time Series',
            ui.nav('ARIMA | SARIMAX',
                   arima_layout),
            ui.nav('Gradient Boosts',
                   gradient_boost_layout),
            '---',
            'Regression | Classification',
            ui.nav('Estimate'),
            ui.nav('Trees'),
            ui.nav('Ensemble'),
            ui.nav('Deep Learning'),
            ),
        title='Model Buddy',
        ),
)

# df = pd.DataFrame()

## Server
def server(input, output, session):
    
    # Create a 'global' dataframe
    @reactive.Calc
    def create_df():
        if input.data_file():
            f: list[FileInfo] = input.data_file()
            df = pd.read_csv(f[0]['datapath'], 
                            sep=input.file_separator(),
                            header=0 if input.file_header() else None)
            return df
        else:
            return pd.DataFrame()

    # Create 'global' time-series dataframe
    @reactive.Calc
    def create_ts_df():
        df = create_df()
        if input.date_column():
            try:
                df['ds'] = pd.to_datetime(df[input.date_column()], format='%Y-%m-%d')
                df.set_index('ds', inplace=True)
            except:
                return 'Error parsing date'
        
        elif input.date_column_gboost():
            try:
                df['ds'] = pd.to_datetime(df[input.date_column_gboost()], format='%Y-%m-%d')
                df.set_index('ds', inplace=True)
            except:
                return 'Error parsing date'
        return df

    @output
    @render.text
    def console_log():
        msg = ''
        if not create_ts_df().empty:
            msg = f'Creating the time-series dataframe:\n\n{create_ts_df()}'
        return msg

    @output
    @render.ui
    def file_overview():
        # if input.data_file():
        df = create_df()
        if input.file_display() == 'Preview':
            return ui.HTML(df.to_html(classes='table table-striped',
                                    max_rows=10)
                            )

    @output
    @render.text
    def file_overview_txt():
        # if input.data_file():
        df = create_df()
        if input.file_display() == 'Structure':
            return f'---Data types---\n\n{df.dtypes}\n\n---Null values---\n\n{df.isna().sum()}'
        elif input.file_display() == 'Summary':
            return f'{df.describe()}'
    
    @output
    @render.data_frame
    def df_grid():
        if input.filter_query():
            df = create_df().query(input.filter_query())
        else:
            df = create_df()
        cols = input.df_columns()
        return render.DataGrid(
            df[[col for col in cols]],
            height=350,
            width='100%',
            # filters=True,
            summary=True
        )

    @output
    @render.ui
    def get_columns():
        df = create_df()
        columns_list = [col for col in df.columns]
        return ui.input_select(
            id='df_columns',
            label='Select column(s) to display:',
            choices=columns_list,
            multiple=True,
            selected=columns_list
        )
    
    @output
    @render.ui
    def get_x():
        df = create_df()
        columns_list = [col for col in df.columns]
        return ui.input_select(
            id='x_values',
            label='Select your x',
            choices=columns_list,
        )
    
    @output
    @render.ui
    def get_y():
        df = create_df()
        columns_list = [col for col in df.columns]
        return ui.input_select(
            id='y_values',
            label='Select your y',
            choices=columns_list,
        )

    @output
    @render_widget
    def plot():
        df = create_df()
        plot_type = input.plot_type()
        if plot_type == 'Line':
            x = df[input.x_values()]
            y = df[input.y_values()]
            title= f'{input.x_values()} Vs. {input.y_values()}'
            return make_lineplot(x,y, title=title)
        elif plot_type == 'Scatter':
            x = df[input.x_values()]
            y = df[input.y_values()]
            title= f'{input.x_values()} Vs. {input.y_values()}'
            return make_scatterplot(x,y, title=title)
        elif plot_type == 'Histogram/Distribution':
            x = df[input.x_values()]
            # y = df[input.y_values()]
            title= f'Distribution of {input.x_values()}'
            return make_hist(df, x, title=title)
        elif plot_type == 'Boxplot':
            # x = df[input.x_values()]
            y = df[input.y_values()]
            title= f'Distribution of {input.y_values()}'
            return make_boxplot(df, y, title=title)
        elif plot_type == 'Bar':
            x = df[input.x_values()]
            y = df[input.y_values()]
            title= f'{input.x_values()} Vs. {input.y_values()}'
            return make_barplot(x,y, title=title)

    @output
    @render.ui
    def get_date_column():
        df = create_df()
        columns_list = [col for col in df.columns]
        return ui.input_select(
            id='date_column',
            label='Select a date column:',
            choices=columns_list,
            selected=''
        )
    
    @output
    @render.ui
    def get_target():
        df = create_df().drop(input.date_column(), axis=1)
        columns_list = [col for col in df.columns]
        return ui.input_select(
            id='target',
            label='Select your target:',
            choices=columns_list
        )

    @output
    @render.ui
    def get_val_decomposition():
        df = create_ts_df().drop(input.date_column(), axis=1)
        if input.decompose_plot():
            columns_list = [col for col in df.columns]
            return ui.input_select(
                id='decompose_val',
                label='Select feature to decompose:',
                choices=columns_list
            )

    @output
    @render_widget
    def model_plot():
        df = create_ts_df()
        if input.decompose_plot() and input.decompose_val():
            x = df[input.decompose_val()]
            return make_decomposition_plots(x)
    
    @output
    @render.ui
    def get_seasonal_hyperpar():
        if input.seasonal():
            return ui.column(6,
                             ui.input_slider(id='p_s', label='Seasonal p', min=0, max=5, value=0),
                             ui.input_slider(id='d_s', label='Seasonal d', min=0, max=5, value=0),
                             ui.input_slider(id='q_s', label='Seasonal q', min=0, max=5, value=0),
                             ui.input_numeric(id='m_s', label='Seasonal period', min=0, max=36, value=0),
                             )
    
    @output
    @render.ui
    def get_exog_feat():
        df = create_ts_df().drop(input.date_column(), axis=1)
        df.drop(input.target(), axis=1, inplace=True)
        if input.exogenous():
            columns_list = [col for col in df.columns]
            return ui.input_select(
                id='exogenous_features',
                label='Select exogenous features:',
                choices=columns_list,
                multiple=True,
                selected=columns_list
            )
    
    @reactive.Calc
    @reactive.event(input.run, ignore_none=False, ignore_init=False)
    def get_arima_results():
        df = create_ts_df()
        target = input.target()

        size = int(len(df) * (1-input.test_size()/100))
        train = df.iloc[:size]
        test = df.iloc[size:]

        order = (input.p(), input.d(), input.q())
        if input.seasonal():
            seasonal_order = (input.p_s(), input.d_s(), input.q_s(), input.m_s())
        else:
            seasonal_order = (0,0,0,0)
        
        if input.exogenous():
            exog_feat = [elem for elem in input.exogenous_features()]
        else:
            exog_feat = None

        return run_arima(train, test, target, order, seasonal_order, exog_feat)

    @output
    # @render_widget()
    @render.plot
    @reactive.event(input.run, ignore_none=False, ignore_init=False)
    def model_results_plot():
        res = get_arima_results()

        plot_train = res['train']
        plot_test = res['test']
        plot_pred = res['prediction']

        return plot_model_results(plot_train, plot_test, plot_pred)
    
    @output
    @render.text
    @reactive.event(input.run, ignore_none=False, ignore_init=False)
    def model_results():

        res = get_arima_results()

        msg = f'''
        --- Dataset Summary ---
        train size: {res['train_size']}
        test size: {res['test_size']}

        --- Model Summary ---
        ARIMA | SARIMAX
        order: {res['order']}
        seasonal order: {res['seasonal_order']}
        exogenous features: {res['exog_feat']}
        
        --- Results ---
        RMSE: {res['rmse']:.2f}
        MAE: {res['mae']:.2f}
        Test mean: {res['mean']:.2f}
        Test std: {res['std']:.2f}
        '''
        return msg
    
    @reactive.Calc
    def create_lagged_df():
        df = create_ts_df()
        if input.lags() == 0:
            return df
        else:
            d = {}
            for col in df.drop([c for c in input.no_lags()], axis=1).columns:
                for i in range(1, input.lags()+1):
                    d[f'{col}-{i}_lag'] = df[col].shift(i)
            tmp = pd.DataFrame.from_dict(d)
            df = pd.concat([df, tmp], axis=1)
            return df.iloc[i:,:]
    
    @output
    @render.text
    def console_log_gboost():
        # msg = ''
        #if not create_lagged_df().empty:
        msg = f'Creating the time-series dataframe with {input.lags()} lags:\n\n{create_lagged_df()}\n\nColumns: {create_lagged_df().columns}'
        return msg
    
    @output
    @render.ui
    def get_no_lags():
        df = create_ts_df()
        cols = [col for col in df.columns]
        return ui.input_select(
            id='no_lags',
            label='Select the columns NOT to transform',
            choices=cols,
            multiple=True,
        )
    
    @reactive.Calc
    @reactive.event(input.run_gboost, ignore_none=False, ignore_init=False)
    def get_gboost_results():
        # if not create_lagged_df().empty:
        #     df = create_lagged_df()
        # else:
        #     df = create_ts_df()
        df = create_lagged_df()
        df.drop(input.date_column_gboost(), axis=1, inplace=False)

        features = [c for c in input.features_gboost()]
        target = input.target_gboost()

        X = df[features]
        y = df[target]

        test_size = input.test_size_gboost()/100

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        if input.gboost_select() == 'XGBoost' and input.reg_cla() == 'Regression':
            model_name = 'XGBoost Regressor'
            params = {
                'learning_rate': input.learning_rate(),
                'n_estimators': input.n_estimators(),
                'max_depth': input.max_depth(),
                'early_stop': input.early_stop()
            }

            print(params)
        else:
            print('Error')

        return run_gboost_model(model_name, X_train, X_test, y_train, y_test, params)

    @output
    @render.ui
    def get_gboost_hyperpar():
        if input.gboost_select() == 'XGBoost':
            res = ui.row(
                    ui.row(
                        ui.column(6,
                                ui.input_numeric(
                                    id='n_estimators',
                                    label='n estimators:',
                                    min=1,
                                    max=50000,
                                    value=10
                                ),
                        ),
                        ui.column(6,
                                ui.input_numeric(
                                    id='learning_rate',
                                    label='learning rate:',
                                    min=0,
                                    max=5,
                                    value=0.1
                                    ),
                        ),
                    ),
                    ui.row(
                        ui.column(6,
                                ui.input_slider(
                                    id='max_depth',
                                    label='max depth:',
                                    min=1,
                                    max=10,
                                    value=2
                                ),
                        ),
                        ui.column(6,
                                ui.input_slider(
                                    id='early_stop',
                                    label='early stopping round:',
                                    min=0,
                                    max=500,
                                    value=100
                                    ),
                        ),
                    ),
                )
        else:
            res = 'Model not available'
    
        return res

    @output
    @render.ui
    def get_date_column_gboost():
        df = create_df()
        columns_list = [col for col in df.columns]
        return ui.input_select(
            id='date_column_gboost',
            label='Select a date column:',
            choices=columns_list,
        )
    
    @output
    @render.ui
    def get_target_gboost():
        df = create_df().drop(input.date_column_gboost(), axis=1)
        columns_list = [col for col in df.columns]
        return ui.input_select(
            id='target_gboost',
            label='Select your target:',
            choices=columns_list
        )
    
    @output
    @render.ui
    def get_features_gboost():
        df = create_lagged_df().drop(input.date_column_gboost(), axis=1)
        # df.drop(input.target_gboost(), axis=1, inplace=True)
        columns_list = [col for col in df.columns]
        return ui.input_select(
            id='features_gboost',
            label='Select features:',
            choices=columns_list,
            multiple=True,
            selected=columns_list
        )

    @output
    # @render_widget()
    @render.plot
    @reactive.event(input.run_gboost, ignore_none=False, ignore_init=False)
    def model_results_plot_gboost():
        res = get_gboost_results()

        plot_train = res['train']
        plot_test = res['test']
        plot_pred = res['prediction']

        return plot_model_results(plot_train, plot_test, plot_pred)
    
    @output
    @render.text
    @reactive.event(input.run_gboost, ignore_none=False, ignore_init=False)
    def model_results_gboost():
        res = get_gboost_results()

        msg = f'''
        --- Dataset Summary ---
        train size: {res['train_size']}
        test size: {res['test_size']}

        --- Model Summary ---
        XGBOOST
        features: {[feat for feat in res['features']]}
        n estimators: {res['params']['n_estimators']}
        learning rate: {res['params']['learning_rate']}
        max depth: {res['params']['max_depth']}
        early stop: {res['params']['early_stop']}

        --- Feature Importance ---
        {res['feature_importance']}
        
        --- Results ---
        RMSE: {res['rmse']:.2f}
        MAE: {res['mae']:.2f}
        Test mean: {res['mean']:.2f}
        Test std: {res['std']:.2f}
        '''
        return msg



# Combine into a shiny app.
# Note that the variable must be "app".
app = App(app_ui, server)