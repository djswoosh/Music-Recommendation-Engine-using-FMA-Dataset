import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from pathlib import Path
import lux

header = st.container()
dataset = st.container()
correlation = st.container()
model_training = st.container()

with header:
	st.title("The Swoosh Streams Music Recommendation Engine")
	st.text('This Data Science project recommends a suggestion of music tracks based on the underlying trends/ or mood of the user/consumer, using content based fitering')

with dataset:
	st.header('The Echo Nest Dataset (Taken from the Free Music Archive)')
	st.text('This Dataset is publically available from the UCI repository')

	swooshstreams = pd.read_csv('swooshstreams.csv')
	st.dataframe(swooshstreams.head())

	st.subheader("Artist Familiarty Distribution")
	artist_familiarity = pd.DataFrame(swooshstreams['artist_familiarity'].value_counts()).head(50)
	st.bar_chart(artist_familiarity)



with correlation:
	def app():
		st.title('Correlation')
		st.write('Check out these correlation charts')
		df = pd.read_csv('swooshstreams.csv')
		export_file = 'swooshstreams.html'
		html_content = df.save_as_html(output=True)
		components.html(html_content, width=800, height=350)

	app()






with model_training:
	st.header("Model training")
	st.text("User can choose their own hyper parameters for the model and see how the performance changes")

	set_col, disp_col = st.columns(2)

	max_depth = set_col.slider('What should be the max_depth of the model ?', min_value=20, max_value=100, value=20, step=10)

	n_estimarots = set_col.selectbox('How many artists should there be?', options  = [100,200,300, 'No limit'], index = 0)

	input_feature = set_col.text_input('Which feature should be used as the input feature?', 'Energy')