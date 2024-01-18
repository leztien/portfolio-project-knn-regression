
"""
Streamlit app for ProteinPathefinders

Assets needed for this to run:
    protein_pathfinders_model_2023_04_26_16_28_33
    protein_pathfinders_model_2023_04_26_18_32_10
    logo.png
    utilities.py
    pateint1.csv ...
"""

from time import sleep
import streamlit as st
from pandas import read_csv, DataFrame, Series, concat
from numpy import nan
from PIL import Image
from utilities import ProteinPathfinders
from warnings import filterwarnings
filterwarnings('ignore', category=UserWarning)



# Load the model
path1 = "protein_pathfinders_model_2023_04_26_16_28_33"  # KPCA
model1 = ProteinPathfinders.from_file(path1)

path2 = "protein_pathfinders_model_2023_04_26_18_32_10"  # KNN +2 features
model2 = ProteinPathfinders.from_file(path2)


# Objects on screen
img = Image.open("logo.png")
st.image(img, width=400)
file = st.file_uploader("")
visit_month = st.select_slider('Number of months since the first visit',
                          ('NA',) + tuple(range(0, 181, 6)))
medication = st.selectbox("Mediaction", ("NA", "Yes", "No"))
button = st.button('Go')


# Control
if file is not None:
    data = read_csv(file).set_index("Unnamed: 0")
    data.index.name = None
    data = data.T
    
    
if button and file:
    if visit_month == 'NA':
        model = model1
    else:
        visit_month = int(visit_month)
        medication = {'NA': nan, 'Yes': 'On', 'No': 'Off'}[medication]
        
        sr0 = Series(visit_month, index=data.index, name="visit_month")
        sr1 = Series(medication, index=data.index, name="upd23b_clinical_state_on_medication")
        data = concat([sr0, data, sr1], axis=1)
        
        model = model2

    # predict
    prediction = list(model.predict(data).round().astype(int).ravel())
    df = DataFrame(prediction, 
                   index=["UPDRS-1", "UPDRS-2", "UPDRS-3", "UPDRS-4"], 
                   columns=["rating"])
    sleep(1)
    st.write("\n\n")
    st.write("**UPDRS (Unified Parkinson's Disease Rating Scale) results:**")
    st.write(df)

    





