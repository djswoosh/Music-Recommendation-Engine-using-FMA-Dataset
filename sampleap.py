import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import pandas as pd
import lux

def app():
    st.title('Generate EDA')
    st.write('Training Data Exploratory Data Analysis ')
    df = pd.read_csv("https://raw.githubusercontent.com/lux-org/lux-datasets/master/data/hpi.csv")
    export_file = 'visualizations.html'
    df.save_as_html(export_file)
    txt = Path(export_file).read_text()
    components.html(txt, width=800, height=350)

app()
