from dash import Dash, html, dcc, callback, Output, Input
from plotly.graph_objs._figure import Figure as PlotlyFigure

app = Dash(__name__)

gantt_chart = dcc.Graph(id='gantt_chart')
gantt_chart.figure = PlotlyFigure()

app.layout = html.Div([
    html.H1(children='Dashboard SimRL', style={'textAlign':'center'}),
    gantt_chart,
    dcc.Interval(
        id='interval-component',
        interval=1*1000, # in milliseconds
        n_intervals=0,
    ),
])

def write_gantt(
    gantt_updated: PlotlyFigure
) -> None:
    gantt_chart.figure = gantt_updated


@callback(
    Output('gantt_chart', 'figure'),
    Input('interval-component', 'n_intervals'),
    prevent_initial_call=True
)
def update_gantt(n_intervals) -> PlotlyFigure:
    #print('++++++++++++++++++++++++++++ Calling Home!!!!!!!!!')
    return gantt_chart.figure


if __name__ == '__main__':
    app.run(debug=True)