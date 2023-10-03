from dash import Dash, html, callback, Output, Input, State, dcc
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import pickle
import numpy as np
import mlflow,os

external_stylesheets = [dbc.themes.LUX]
app = Dash(__name__, external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
model_name = os.environ['APP_MODEL_NAME']
model_version = 1

# load all components 
loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
loaded_encoder = pickle.load(open('./model/label_encoderA3.model', 'rb'))
loaded_scaler = pickle.load(open('./model/minmax_scalerA3.model', 'rb'))

# Create the function for gathering all features and preparing them for the prediction
def get_features(MAXPOWER,YEAR,selected_FUEL,selected_brand):
    #Scale features: 'max_power','year'
    sample = pd.DataFrame([[float(MAXPOWER),float(YEAR), str(selected_FUEL),str(selected_brand)]],columns=['max_power','year','fuel','brand'])
    sample[['max_power','year']]  = loaded_scaler.transform(sample[['max_power','year']])
    #Lable encoding 'fuel' before prediction
    sample['fuel'] = loaded_encoder.transform(sample['fuel'])
    # Onehot encoding 'brand' before prediction
    sample_encoded =onehot(sample,'brand')
    # Return clean numpy array
    return sample_encoded.to_numpy()

# Predict the price
def price_prediction(MAXPOWER,YEAR,selected_FUEL,selected_brand):
    #Get the features array
    sample_encoded = get_features(MAXPOWER,YEAR,selected_FUEL,selected_brand)
    #Predicted the price by loading model through the mlflow server
    predicted = loaded_model.predict(sample_encoded)
    return  predicted

# Create the one-hot encoder function to use in get_features function.
def onehot(dataframe, column_name):
    test = dataframe
    # list of brands
    brands = ['Ambassador', 'Ashok', 'Audi', 'BMW', 'Chevrolet', 'Daewoo', 'Datsun', 'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo']
    for brand in brands:
        if brand == test[column_name][0]:
            test[brand] = 1
        else:
            test[brand] = 0
    test.drop(columns=column_name,inplace = True)
    return test



# Define the layout for the Main Page
main_layout = html.Div([
    
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Logistic Regression Model", href="/logistic_model")),
        ],
        brand="Chacky Company Co., Ltd.",
        brand_href="/",
        color="primary",
        dark=True,
    ),
    
    html.Div(id="page-content")
],
    style={"text-align": "center"})

# Set up URL routing
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    main_layout
],
    style={"text-align": "center"})

intro_layout=  html.Div([
    html.Img(src='/assets/Car-Prices-1068x601.jpg', style={'display': 'block', 'margin': '0 auto','width': '40%', 'height': 'auto'}),
    html.Br(),
    html.Label('CHACKY COMPANY CO., LTD.', style={'textAlign': 'center', 'fontSize': 80, 'marginTop': '5vh',"font-weight": "1000"}),
    html.Label( ' You are now entering: Car\'s Price Prediction site', style={'textAlign': 'center', 'fontSize': 40,"font-weight": "400"}),
    html.Br(),
    html.Br(),
    html.Div([
        html.Label( "Welcome to Chacky Company Co., Ltd. We are the carmaker but we do not know how to set the car prices. \
               So we came up with the idea of hiring the outsource which is a master's degree student to create the prediction machine learning model.   \
               The past machine learning models are linear regression and linear regression from scratch, the results are satisfied for the regression problems. \
               But, we are not done yet. We want to deal with classification problems that lead to a load of work for that student.", style={'textAlign': 'justify', 'fontSize': 20,"font-weight": "200"}),
        html.Br(),
        html.Br(),
        html.Label( "Recently, our company launched a new prediction website that helps the company define their car prices in 4 categories: 'Low price', 'Good price', 'Great price', and 'Expensive price'. The predicted model is based on the Logistic Regression integrated with the Ridge regularization. \
                The model have 72.4 percent accuracy with hyperparameters as follows: Normal regularization, method=batch, alpha=0.001, and max_iter=10000. ", style={'textAlign': 'justify', 'fontSize': 20,"font-weight": "200"}),
        html.Br(),
        html.Br(),
        html.Label("To access the model, click the right-upper navigation link 'Logistic Regression model'.", style={'textAlign': 'center', 'fontSize': 20,"font-weight": "800"})
    ], style={'margin-left': '10%', 'margin-right': '10%', 'margin-bottom': '10vh'})
    
])
# Callback to update page content based on the URL
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/logistic_model":
        return content_layout
    elif pathname == "/":
        return intro_layout

content_layout= dbc.Container([
    dbc.Row([
        html.Div([
            html.Br(),
            html.Label("Car's price prediction (Classification Version)",  style={"font-size": "50px", "font-weight": "Bold", "vertical-align" : "center",'color':'Blue'}),
            html.Div([
                html.Label( "For instruction, users need to define 4 features: the car's brand, the maximum power of the car (Horsepower), the year of the car (AD), and the fuel type of the car. \
                           After selecting or defining all features, click the 'PREDICTION' button below to submit all of the inputs. \
                            Then the result will show the appropriate setting price in the form of classification which are: 'Low price', 'Good price', 'Great price', and 'Expensive price'.", 
                            style={'textAlign': 'justify', "font-size": "20px","font-weight": "200"}),
                html.Br(),
                html.Br(),
                html.Label("Remarks: There are default values in each feature in case the user does not type or select those inputs. ",style={'textAlign': 'justify', "font-size": "20px","font-weight": "1000"})
                             ], style={'margin-left': '10%', 'margin-right': '10%'}),
                html.Label("[Brand = 'Toyota', Max_power = 100.0, Year = 2010, Fuel = 'Petrol']",style={'textAlign': 'justify', "font-size": "15px","font-weight": "800"}),
            html.Br(),
            html.Br(),
                html.Label("======= Let's predict your car prices! ======="
                        ,  style={"font-size": "40px", "font-weight": "Bold","vertical-align" : "center",'color':'Navy'}),
            html.Br(),
            html.Br(),
            html.Div([
                dbc.Label("Please select brand of the car.", style={"font-size": "25px"}),
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
                html.Br(),
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
                html.Button(id="SUBMIT", children="Prediction", className="btn btn-outline-primary"), 
                html.Br(),
                html.Br(),
                dbc.Card([
                    dbc.CardHeader("Predicted category of price",style={"font-size": "30px", "font-weight": "700", "color":"white"}),
                    dbc.CardBody(html.Output(id="selling",style={"font-size": "30px", "font-weight": "700", "color":"white"}))
                    ],style={   
                                "display": "flex",
                                "justify-content": "center",  
                                "align-items": "center",      
                                "width": "500px",             
                                "margin": "auto",           
                            }, color="info", inverse=True)], style={'margin-left': '10%', 'margin-right': '10%'})
        ],
        className="mb-3")
    ],justify='center')

], fluid=True)
@callback(
    Output(component_id="selling", component_property="children"),
    State(component_id="MAXPOWER", component_property="value"),
    State(component_id="YEAR", component_property="value"),
    State(component_id="selected_FUEL",component_property='value'),
    State(component_id="selected_brand", component_property="value"),
    Input(component_id="SUBMIT", component_property='n_clicks'),
    
    prevent_initial_call=True
)


# Mapping the results
def Prediction(MAXPOWER,YEAR,selected_FUEL,selected_brand,SUBMIT):
    # default values
    dict_default={"MAXPOWER": 100.0,"YEAR":2010,"selected_FUEL":"Petrol","selected_brand":'Toyota'}
    if MAXPOWER is None:
        MAXPOWER = dict_default["MAXPOWER"]
    if YEAR is None:
        YEAR = dict_default["YEAR"]
    if selected_FUEL is None:
        selected_FUEL = dict_default["selected_FUEL"]
    if selected_brand is None:
        selected_brand = dict_default["selected_brand"]
    # Get the predicted value
    result = price_prediction(MAXPOWER,YEAR,selected_FUEL,selected_brand)
    
    # Create the map
    price_mapping = {
    0: 'Low price',
    1: 'Good price',
    2: 'Great price',
    3: 'Expensive price'
    }
    # Map the price
    mapped_value = price_mapping.get(result[0], 'Unknown price')
    return mapped_value




if __name__ == "__main__":
    app.run_server(debug=True)
