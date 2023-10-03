from shiny import ui
from shinywidgets import output_widget

layout = ui.layout_sidebar(
            ui.panel_sidebar(
                ui.panel_well(
                    ui.input_file(id='data_file',
                                    label='Load data from:',
                                    accept='.csv'),
                ),
                
                ui.panel_well(
                    ui.tags.h5('Overview:'),
                    ui.tags.br(),
                    ui.row(
                        ui.column(6, 
                                  ui.row(ui.input_checkbox(id='file_header',
                                            label='Header',
                                            value=True)
                                            ),
                                  ui.row(ui.input_select(id='file_separator',
                                            label='Separator:',
                                            selected='Comma',
                                            choices={',':'Comma', ';':'Semicolon', '\t':'Tab'})
                                            )
                                    ),
                        ui.column(6, ui.input_radio_buttons(
                                    id='file_display',
                                    label='Visualize:',
                                    selected='Preview',
                                    choices=['Preview', 'Structure', 'Summary']
                                )
                            ),
                        ),
                ),
                ui.panel_well(
                    ui.tags.h5('Filter:'),
                    ui.tags.br(),
                    ui.row(
                        ui.input_text_area(id='filter_query',
                                           label='',
                                           placeholder='e.g.: cases > 2000'),
                        ui.output_ui(id='get_columns'),
                    )
                ),
                ui.panel_well(
                    ui.tags.h5('Plot:'),
                    ui.tags.br(),
                    ui.row(
                        ui.column(6, 
                                  ui.output_ui(id='get_x')
                                    ),
                        ui.column(6, 
                                  ui.output_ui(id='get_y')
                                ),
                        ),
                    
                    ui.row(
                        ui.input_select(id='plot_type',
                                        label='Choose a type of plot',
                                        choices=['Line', 'Bar', 'Histogram/Distribution', 'Boxplot', 'Scatter']
                                        ),
                    )
                ),
            ),

            ui.panel_main(
                ui.navset_tab_card(
                    ui.nav('Overview',
                            ui.output_ui(id='file_overview'),
                            ui.output_text_verbatim(id='file_overview_txt')
                            ),
                    ui.nav('Filter',
                            ui.output_data_frame(id='df_grid'),
                            ),
                    ui.nav('Plot',
                           output_widget(id='plot')
                           ),
                ),
            ),
        )   