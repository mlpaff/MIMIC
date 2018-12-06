import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event

from plotly import graph_objs as go
from datetime import datetime as dt
import os
import pickle as pk
import pandas as pd
import base64

from gensim.models import Word2Vec
import re
import nltk
import string
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

##### Import data and model #####

## Model
filename = './models/nlp_model.pkl'
with open(filename, 'rb') as inFile:
    lr_model = pk.load(inFile)

## Data
filename = 'data/adm_table.pkl'
adm_data = pd.read_pickle(filename)

## Word2Vec model 
model = Word2Vec.load('mimic_w2v_model.bin')

## Nano Image
image_nano = 'nano-logo.svg'
encoded_nano = base64.b64encode(open(image_nano, 'rb').read())

#### Create options for patient ID Dropdown - This will probably be replaced in the future
patients = adm_data.hadm_id

app = dash.Dash()

app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
# external_css = ["https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
#                 "https://cdn.rawgit.com/plotly/dash-app-stylesheets/737dc4ab11f7a1a8d6b5645d26f69133d97062ae/dash-wind-streaming.css",
#                 "https://fonts.googleapis.com/css?family=Raleway:400,400i,700,700i",
#                 "https://fonts.googleapis.com/css?family=Product+Sans:400,400i,700,700i"]


# Suppress callback exception warnings
app.config.suppress_callback_exceptions = True

## Page layout
app.layout = html.Div(children=[

    ## Title ##
    html.Div([
        html.Div([
            html.A([html.Img(src='data:image/svg+xml;base64,{}'.format(encoded_nano.decode()),
                style={'height': '50px',
                       'float': 'left'})
            ], href='https://nanovision.com')
        ],
        className='four columns'
        )
    ],
    className='row',
    style={'background-color': '#1b0066',
           'padding': '5px'}
    ),

    # html.Div([
    #     html.Div([
    #         html.H3("placeholder")
    #     ], className='Title')
    # ],
    # className='row wind-speed-row'
    # ),
    html.Div([
            html.H3('Patient Discharge Summary',
                style={'text-align': 'center',
                'color': 'White'})
        ],
        className='Title'
        ),

    ## patient input-output ##
    html.Div(children=[
        ## Patient parameter entry ##
        html.Div([

            ## Select Discharge Date
            html.Div([
                html.P('Discharge Date: ',
                    style={'color': 'White'}),
            	dcc.DatePickerSingle(
            		id='my-date-picker-single',
            		# min_date_allowed=dt(2018, 6, 4),
            		initial_visible_month=dt(2018, 1, 1),
            		date=dt.now(),
            		display_format='MMM Do, YYYY'
            	)
            ]
            ),

            #Input for patient name
            html.Div([
                html.Br(),
                html.P('Patient Name: ',
                    style={'color': 'White'}),
                dcc.Input(
                    id='pat-name',
                    placeholder='Enter patient name...',
                    style={'width': '50%',
                            'height': 25}
                ),
                html.Br()
            ],
            #style={'width': '45%'}
            ),


            #Input for patient name
            html.Div([
                html.Br(),
                html.P('Patient Age: ',
                    style={'color': 'White'}),
                dcc.Input(
                    id='pat-age',
                    placeholder='Enter patient age...',
                    style={'width': '50%',
                            'height': 25}
                ),
                html.Br()
            ],
            #style={'width': '45%'}
            ),

            ## List of diagnoses IDs
            html.Div([
                html.Br(),
                html.P('Diagnosis IDs:',
                    style={'color': 'White'}
                ),
                dcc.Textarea(
                    id='diagnosis',
                    placeholder='Enter diagnosis codes...',
                    style={'width': '100%'}
                ),
                html.Br()
            ],
            # style={'width': '100%'}
            ),
        ],
        className='six columns data-entry',
        style={
            'background-color':'#1b0066',
            'padding': '10px'}
        ),

        ## Risk Output ##
        html.Div(children=[
            # html.Div([
            #     html.H3('Risk of HAI',
            #         style={'text-align': 'center'})
            # ],
            # className='Title'
            # ),
            #Searchable dropdown for patient ID --- This is for prototype ---
            html.Div([
                html.P('Patient ID: ',
                    style={'color': 'White'}),
                dcc.Dropdown(
                    id='pat-id',
                    placeholder='Select patient ID...',
                    options=[{'label': str(num), 'value': str(num)} for num in patients],
                    searchable=True
                ),
                html.Br()
            ],
            #style={'width': '45%'}
            ),

            ## Patient Insurance
            html.Div([
                html.P('Insurance:',
                    style={'color': 'White'}),
                dcc.Dropdown(
                    id='insurance',
                    options=[
                        {'label': 'Medicare', 'value': 'medicare'},
                        {'label': 'Private', 'value': 'private'},
                        {'label': 'Medicaid', 'value': 'medicaid'},
                        {'label': 'Government', 'value': 'gov'},
                        {'label': 'Self Pay', 'value': 'selfpay'},
                    ],
                    searchable=False
                ),
                html.Br()
            ],
            #style={'width': '45%'}
            ),

            ## Admission type
            html.Div([
                html.P('Admission Type:',
                    style={'color': 'White'}),
                dcc.Dropdown(
                    id='adm_type',
                    options=[
                        {'label': 'Emergency', 'value': 'emergency'},
                        {'label': 'Elective', 'value': 'elective'},
                        {'label': 'Newborn', 'value': 'newborn'},
                        {'label': 'Urgent', 'value': 'urgent'}
                    ],
                    searchable=False
                ),
                html.Br()
            ],
            #style={'width': '45%'}
            ),

            html.Div([
                html.P('Discharge Notes:',
                    style={'color': 'White'}
                ),
                dcc.Textarea(
                    id='discharge',
                    placeholder='Enter discharge notes...',
                    style={'width': '100%'}
                )
            ]
            ),

            ## Button to calculate risk
            html.Div([
                html.Br(),
                html.Button(
                'Calculate Readmission Risk',
                id='my-button',
                style={'background-color': 'LightGrey'})
            ]
            ),
        ],
        className='six columns output',
        style={
            'height': '450px',
            # 'color':'White',
            'padding': '10px'}
        )
    ],
    className='row main-input'
    ),

    html.Div(children=[
            
            html.Div([
                html.Br(style={'line-height': '55px'}),
                html.Div(['Enter patient data for readmission risk evaluation...'
                ],
                id='pred_score',
                style={'fontSize': '24px'}
                )
            ])
    ],
    style={
        'color':'White',
        'padding': '10px'},
    className='row ouput'
    )

],
style={
    'padding': '0px 10px 10px 10px',
    'marginLeft': 'auto',
    'marginRight': 'auto',
    'background-color': '#1b0066',
    "width": "800px",
    'boxShadow': '0px 0px 15px 15px rgba(204,204,204,0.4)'}
)

## Given patient ID, predict probability of HAI
# @app.callback(
#     Output('pred_score', 'children'),
#     [Input('pat-id', 'value'),
#      Input('insurance', 'value'),
#      Input('adm_type', 'value'),
#      Input('marital', 'value'),
#      Input('button', 'n_clicks')],
#     [State('pat-id', 'value'),
#      State('insurance', 'value')])

@app.callback(
    Output('pred_score', 'children'),
    [Input('my-button', 'n_clicks')],
    [State('pat-name', 'value'),
     State('pat-id', 'value'),
     State('insurance', 'value'),
     State('adm_type', 'value'),
     State('diagnosis', 'value'),
     State('discharge', 'value')]
)
def predict_hai_risk(n_clicks, pat_name, pat_id, ins, adm_type, diagnosis, discharge):
    if n_clicks > 0 and (pat_name == None or pat_id==None):
        return 'Please complete data entry...'
    if pat_id == None:
        return 'Enter patient data for readmission risk evaluation...'
    prob = 40
    # pat = test_data[test_data['subject_id'] == int(pat_id)]
    # pat = pat.drop('subject_id', axis = 1)
    # prob = lr_model.predict_proba(pat)[0][1] * 100
    if prob >= 30:
        return '{} has a {}% risk for developing a complication within 30 days. Recommend scheduling follow up with primary care physician within 2 weeks'.format(pat_name, round(prob, 2))
    else:
        return '{} is NOT at risk for developing complications within 30 days.'.format(pat_name)

    # if prob >= 35:
    #     return 'Patient has a {}% risk of HAI. Take extra precautions when treating.'.format(round(prob, 2))
    # else:
    #     return 'Patient has a {}% risk of HAI.'.format(round(prob, 2))

# if 'DYNO' in os.environ:
#     app.scripts.append_script({
#         'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
#     })

if __name__ == '__main__':
	app.run_server(debug=True)
