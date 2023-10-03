from shiny import ui
from shinywidgets import output_widget

arima_layout = ui.layout_sidebar(
                    ui.panel_sidebar(
                        'ARIMA | SARIMAX Model',
                        ui.panel_well(
                            ui.tags.h5('General:'),
                            ui.tags.br(),
                            ui.row(
                                ui.column(6, ui.output_ui(id='get_date_column')),
                                ui.column(6, ui.output_ui(id='get_target'))
                            ),
                            ui.row(
                                ui.input_slider(id='test_size',
                                                label='Select test size (%):',
                                                min=0,
                                                max=100,
                                                value=20)
                            )
                        ),
                        ui.panel_well(
                            ui.tags.h5('Plot:'),
                            ui.tags.br(),
                            ui.row(
                                ui.input_checkbox(id='decompose_plot',
                                                  label='Make a decomposition plot'
                                                  ),
                                ui.output_ui(id='get_val_decomposition')
                                ),
                        ),
                        ui.panel_well(
                            ui.tags.h5('Hyperparameters:'),
                            ui.tags.br(),
                            ui.row(
                                ui.column(6,
                                          ui.input_checkbox(id='seasonal',
                                                            label='Is seasonal?')
                                          ),
                                ui.column(6,
                                          ui.input_checkbox(id='exogenous',
                                                            label='Use exogenous features?')
                                          )
                            ),
                            ui.row(
                                ui.column(6,
                                          ui.input_slider(id='p', label='Autoregressive (p)', min=0, max=5, value=0),
                                          ui.input_slider(id='d', label='Differential (d)', min=0, max=5, value=0),
                                          ui.input_slider(id='q', label='Moving average (q)', min=0, max=5, value=0)
                                          ),
                                ui.column(6,ui.output_ui(id='get_seasonal_hyperpar'))
                            ),
                            ui.row(
                                ui.tags.p('Search for best hyperparameters (RMSE):'),
                                ui.input_action_button(id='hyper_search',
                                                       label='Search',
                                                       width='100px',
                                                       class_='btn-success')
                            ),
                            ui.row(
                                ui.output_ui(id='get_exog_feat')
                            ),
                        ),
                        ui.panel_well(
                            ui.row(
                                    ui.column(6, ui.input_action_button(id='run',
                                                                        label='Run')),
                                    ui.column(6, ui.input_action_button(id='save_gboost',
                                                                        label='Save model',
                                                                        class_='btn-success'))
                                    ),
                            # ui.input_action_button(id='run',
                            #                         label='Run')
                        )
                    ),

                    ui.panel_main(
                        ui.navset_tab_card(
                            ui.nav('Console',
                                   ui.output_text_verbatim(id='console_log'),
                                    ),
                            ui.nav('Plot',
                                   output_widget(id='model_plot')
                                    ),
                            ui.nav('Model results',
                                   ui.output_plot(id='model_results_plot'),
                                   ui.output_text_verbatim(id='model_results')
                                   ),
                        ),
                    ),
                )


gradient_boost_layout = ui.layout_sidebar(
                            ui.panel_sidebar(
                                'Gradient Boosts | Time-series',
                                ui.panel_well(
                                    ui.tags.h5('General:'),
                                    ui.tags.br(),
                                    ui.row(
                                        ui.column(6, ui.output_ui(id='get_date_column_gboost')),
                                        ui.column(6, ui.output_ui(id='get_target_gboost'))
                                    ),
                                    ui.row(
                                        ui.column(6,
                                                  ui.input_radio_buttons(id='gboost_select',
                                                               label='Select the model:',
                                                               choices=['XGBoost', 'Catboost', 'LightGBM'],
                                                               selected='XGBoost')
                                                  ),
                                        ui.column(6,
                                                  ui.input_radio_buttons(id='reg_cla',
                                                               label='Type of model:',
                                                               choices=['Regression', 'Classification'],
                                                               selected='Regression')
                                                  ),
                                    ),
                                    ui.row(
                                        ui.input_slider(id='test_size_gboost',
                                                        label='Select test size (%):',
                                                        min=0,
                                                        max=100,
                                                        value=20)
                                    )
                                ),
                                ui.panel_well(
                                    ui.tags.h5('Feature Engineering:'),
                                    ui.tags.br(),
                                    ui.row(
                                        ui.column(6,
                                                  ui.input_slider(id='lags',
                                                        label='Select the number of lags:',
                                                        min=0,
                                                        max=5,
                                                        value=0
                                                        ),
                                                    ),
                                            
                                        ui.column(6,
                                                  ui.output_ui(id='get_no_lags'),
                                                  ),
                                        ),  
                                    ui.row(ui.output_ui(id='get_features_gboost')),
                                ),
                                ui.panel_well(
                                    ui.tags.h5('Hyperparameters:'),
                                    ui.tags.br(),
                                    ui.row(
                                        ui.output_ui(id='get_gboost_hyperpar'),
                                    ),
                                ),
                                ui.panel_well(
                                    ui.row(
                                        ui.column(6, ui.input_action_button(id='run_gboost',
                                                                            label='Run')),
                                        ui.column(6, ui.input_action_button(id='save_gboost',
                                                                            label='Save model',
                                                                            class_='btn-success'))
                                    ),
                                    
                                )
                            ),

                            ui.panel_main(
                                ui.navset_tab_card(
                                    ui.nav('Console',
                                        ui.output_text_verbatim(id='console_log_gboost'),
                                            ),
                                    # ui.nav('Plot',
                                    #     output_widget(id='model_plot')
                                    #         ),
                                    ui.nav('Model results',
                                        ui.output_plot(id='model_results_plot_gboost'),
                                        ui.output_text_verbatim(id='model_results_gboost')
                                        ),
                                ),
                            ),
                        )