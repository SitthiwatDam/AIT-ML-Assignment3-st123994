
# Import packages
from dash import Dash, html, callback, Output, Input, State, dcc
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import pickle
import numpy as np
import mlflow
import os
# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.LUX]
a3_model = Dash(__name__, external_stylesheets=external_stylesheets)

mlflow.set_tracking_uri('https://mlflow.cs.ait.ac.th')
model_name = os.environ['APP_MODEL_NAME']
model_version = 1

# load all components 
# loaded_model = pickle.load(open('./model/model.pkl', 'rb'))
# loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
loaded_model= mlflow.pyfunc.load_model(model_uri=f"models:/st123994-a3-model/1")
loaded_encoder = pickle.load(open('./model/label_encoderA3.model', 'rb'))
# loaded_onehot = pickle.load(open('./model/onehot_encoderA3.model', 'rb'))
loaded_scaler = pickle.load(open('./model/minmax_scalerA3.model', 'rb'))


# App layout
a3_model.layout = dbc.Container([
    dbc.Row([
        html.Div([
            html.Br(),
            
            html.Label("-------Car's price prediction (Newest-Version)-------",  style={"font-size": "50px", "font-weight": "Bold", "vertical-align" : "center",'color':'Blue'}),
            html.Br(),
            html.Label("description"
                    ,  style={"font-size": "20px", "vertical-align" : "center",'color':'black'}),
            html.Br(),
                html.Label("Let's predict your car prices!"
                        ,  style={"font-size": "20px", "vertical-align" : "center",'color':'black'}),
            html.Br(),
            html.Br(),
            dbc.Label("Please select the brand.", style={"font-size": "25px"}),
            dbc.Select(id='selected_brand', size="lg", options=[
                {"label": "Ambassador", "value": "Ambassador"},
                {"label": "Ashok", "value": "Ashok"},
                {"label": "Audi", "value": "Audi"},
                {"label": "BMW", "value": "BMW"},
                {"label": "Chevrolet", "value": "Chevrolet"},
                {"label": "Daewoo", "value": "Daewoo"},
                {"label": "Datsun", "value": "Datsun"},
                {"label": "Fiat", "value": "Fiat"},
                {"label": "Force", "value": "Force"},
                {"label": "Ford", "value": "Ford"},
                {"label": "Honda", "value": "Honda"},
                {"label": "Hyundai", "value": "Hyundai"},
                {"label": "Isuzu", "value": "Isuzu"},
                {"label": "Jaguar", "value": "Jaguar"},
                {"label": "Jeep", "value": "Jeep"},
                {"label": "Kia", "value": "Kia"},
                {"label": "LandRover", "value": "Land"},
                {"label": "Lexus", "value": "Lexus"},
                {"label": "MG", "value": "MG"},
                {"label": "Mahindra", "value": "Mahindra"},
                {"label": "Maruti", "value": "Maruti"},
                {"label": "Mercedes-Benz", "value": "Mercedes-Benz"},
                {"label": "Mitsubishi", "value": "Mitsubishi"},
                {"label": "Nissan", "value": "Nissan"},
                {"label": "Opel", "value": "Opel"},
                {"label": "Peugeot", "value": "Peugeot"},
                {"label": "Renault", "value": "Renault"},
                {"label": "Skoda", "value": "Skoda"},
                {"label": "Tata", "value": "Tata"},
                {"label": "Toyota", "value": "Toyota"},
                {"label": "Volkswagen", "value": "Volkswagen"},
                {"label": "Volvo", "value": "Volvo"},
                ]
            ),
            dbc.Label("Type the max power of the car.",  style={"font-size": "25px"}),
            dbc.Input(id="MAXPOWER", type="number", placeholder="Put a value for max power", size="lg"),
            html.Br(),
            dbc.Label("Type the year of the car.", style={"font-size": "25px", 'text-align' : 'middle'}),
            dbc.Input(id="YEAR", type="number", placeholder="Put a value for year", size="lg"),
            html.Br(),
            dbc.Label("Please select the type of fuel.", style={"font-size": "25px"}),
            dbc.Select(id='selected_FUEL', size="lg", options=[
                {"label": "Diesel", "value": "Diesel"},
                {"label": "Petrol", "value": "Petrol"},]),
            
            html.Br(),
            html.Button(id="SUBMIT", children="price_prediction", className="btn btn-outline-primary"), 
            html.Br(),
            html.Br(),
            dbc.Card([
                    dbc.CardHeader("Predicted price (Baht)",style={"font-size": "30px", "font-weight": "700", "color":"black"}),
                    dbc.CardBody(html.Output(id="selling",style={"font-size": "30px", "font-weight": "700", "color":"black"}))
                    ],style={   
                                "display": "flex",
                                "justify-content": "center",  
                                "align-items": "center",      
                                "width": "500px",             
                                "margin": "auto",           
                            }, color="info", inverse=True)
            
        ],
        className="mb-3")
    ],justify='center')

], fluid=True)


@callback(
    Output(component_id="selling", component_property="children"),
    State(component_id="selected_brand", component_property="value"),
    State(component_id="MAXPOWER", component_property="value"),
    State(component_id="YEAR", component_property="value"),
    State(component_id="selected_FUEL",component_property='value'),
    Input(component_id="SUBMIT", component_property='n_clicks'),
    
    prevent_initial_call=True
)



def price_prediction(MAXPOWER,YEAR,selected_FUEL,SUBMIT,selected_brand):
    
    #Scale features: 'max_power','year'
    sample = pd.DataFrame([[float(MAXPOWER),float(YEAR), str(selected_FUEL),str(selected_brand)]],columns=['max_power','year','fuel','brand'])
    sample[['max_power','year']]  = loaded_scaler.transform(sample[['max_power','year']])
    
    #Lable encoding 'fuel' before prediction
    sample['fuel'] = loaded_encoder.transform(sample['fuel'])

    # Onehot encoding 'brand' before prediction
    sample_encoded =onehot(sample,'brand')

    #Predicted the price by loading model that we saved
    predicted_exp = loaded_model.predict(sample_encoded)
    
    return  dbc.Col(dbc.Card(predicted_exp  , color="info", inverse=True))

def onehot(dataframe, column_name):
    test = dataframe
    brands = ['Ambassador', 'Ashok', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo']
    for brand in brands:
        if brand == test[column_name][0]:
            test[brand] = 1
        else:
            test[brand] = 0
    test.drop(columns=column_name,inplace = True)
    return test

