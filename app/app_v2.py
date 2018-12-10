import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event

from plotly import graph_objs as go
from datetime import datetime as dt
import os
import pickle as pk
import pandas as pd
import numpy as np
import base64

from gensim.models import Word2Vec, KeyedVectors
import nlpHelpers as H
import nltk

##### Import data and model #####

## Model
filename = '../models/nlp_model.pkl'
with open(filename, 'rb') as inFile:
    lr_model = pk.load(inFile)

## Data
filename = '../data/adm_table.pkl'
adm_data = pd.read_pickle(filename)

## Word2Vec model 
w2vModel = KeyedVectors.load_word2vec_format('../models/mimic_w2v_model.bin', binary=True)
# Get vectorizer
w2vVectorizer = H.MeanEmbeddingVectorizer(w2vModel)

## Nano Image
image_nano = 'nano-logo.svg'
encoded_nano = base64.b64encode(open(image_nano, 'rb').read())

## Hard-coded feature names
feature_set_1 = ['admission_type', 'total_prior_admits','gender', 'age', 'length_of_stay', \
                 'num_medications', 'num_lab_tests', 'perc_tests_abnormal', 'num_diagnosis']

#### Create options for patient ID Dropdown - This will probably be replaced in the future
patients = adm_data.hadm_id

app = dash.Dash(__name__)

# app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
# app.css.append_css({'relative_package_path'}).append('./stylesheet.css')


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

    ## Title ##
    html.Div([
            html.H2('Patient Discharge Summary',
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
            		display_format='MMM Do, YYYY',
            	)
            ],
            style={'padding-bottom': '15px'}
            ),  

            #Input for patient name
            html.Div([
                # html.Br(),
                html.P('Patient Name: ',
                    style={'color': 'White'}),
                dcc.Input(
                    id='pat-name',
                    placeholder='Enter patient name...',
                    style={'width': '50%',
                            'height': '36px',
                            'font-size': '14px',
                            'padding-left': '7px'}
                ),
            ],
            style={'padding-bottom': '15px'}
            ),

            #Input for patient name
            html.Div([
                # html.Br(),
                html.P('Patient Age: ',
                    style={'color': 'White'}),
                dcc.Input(
                    id='pat-age',
                    placeholder='Enter patient age...',
                    style={'width': '50%',
                            'height': '36px',
                            'font-size': '14px',
                            'padding-left': '7px'}
                ),
            ],
            style={'padding-bottom': '15px'}
            ),
        ],
        className='six columns data-entry',
        style={
            'background-color':'#1b0066',
            'padding-left': '10px',
            'padding-right': '10px'}
        ),

        ## Second column input ##
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
                # html.Br()
            ],
            style={'padding-bottom': '15px',
                   'width': '96%'
            }
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
                # html.Br()
            ],
            style={'padding-bottom': '15px',
                    'width': '96%'
            }
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
                    searchable=False,
                ),
            ],
            style={'padding-bottom': '15px',
                    'width': '96%'
            }
            ),
        ],
        className='six columns output',
        # style={
        #     'padding': '15px'}
        )
    ],
    className='row main-input',
    ),

    html.Div(children=[

        html.Div([
            html.P('Discharge Notes:',
                style={'color': 'White',
                        'font-size': '22px'
                }
            ),
            dcc.Textarea(
                id='discharge',
                placeholder='Enter discharge notes...',
                style={'width': '100%',
                    'height': '150px'
                }
            )
        ]
        ),

        ## Button to calculate risk
        html.Div([
            # html.Br(),
            html.Button(
            'Calculate Readmission Risk',
            id='my-button',
            style={'background-color': 'LightGrey'})
        ],
        style={'padding-top': '15px'}
        ),
    ],
    className='row notes-input',
    style={
        'background-color':'#1b0066',
        'padding-left': '10px',
        'padding-right': '10px',
        'font-size': 20}
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
    'font-size': 22,
    'background-color': '#1b0066',
    "width": "800px",
    'boxShadow': '0px 0px 15px 15px rgba(204,204,204,0.4)'}
)

## Given patient ID, predict probability of readmission
@app.callback(
    Output('pred_score', 'children'),
    [Input('my-button', 'n_clicks')],
    [State('pat-name', 'value'),
     State('pat-id', 'value'),
     State('insurance', 'value'),
     State('adm_type', 'value'),
     State('discharge', 'value')]
)
def predict_hai_risk(n_clicks, pat_name, pat_id, ins, adm_type, discharge):
    if n_clicks > 0 and (pat_name == None or pat_id==None):
        return 'Please complete data entry...'
    if pat_id == None:
        return 'Enter patient data for readmission risk evaluation...'
    else:
        pat_id = int(pat_id)

    note = discharge.replace("\\n", " ")

    note_vector = w2vVectorizer.vectorizeSingleNote(note)

    
    pat = H.prepPatient(adm_data, pat_id, note_vector, feature_set_1)
    prob = lr_model.predict_proba([pat])[0][1] * 100
    
    if prob >= 40:
        return '{} is at a {}% risk of returning with an unexpected complication within 30 days. Recommend scheduling follow up with primary care physician within 2 weeks'.format(pat_name, round(prob, 2))
    else:
        return '{} is NOT at risk for requiring unexpected readmission within 30 days.'.format(pat_name)

if __name__ == '__main__':
	app.run_server(debug=True)
