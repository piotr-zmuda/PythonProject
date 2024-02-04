import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import statsmodels.formula.api as smf
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

st.set_page_config(layout = "wide")


base_data = pd.read_csv("messy_data.csv", delimiter=',')
base_data.replace({" " : np.nan}, inplace=True)
base_data.columns = base_data.columns.str.strip()

base_data['clarity', 'color', 'cut'] = base_data['clarity', 'color', 'cut'].apply(lambda x: x.str.lower())

numeric_columns = ['carat', 'x dimension', 'y dimension', 'z dimension', 'depth', 'table', 'price']
base_data[numeric_columns] = base_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

base_data_filled = base_data.copy()
base_data['table'] = pd.to_numeric(base_data['table'], errors='coerce')
mean_value = pd.to_numeric(base_data['table'], errors='coerce').mean(skipna=True)
print(mean_value)
base_data['table'].fillna(mean_value, inplace=True)
base_data['table'] = base_data['table'].astype(int)

base_data['clarity'].fillna('Other', inplace=True)
base_data['clarity'] = base_data['clarity'].astype(str)

base_data['color'].fillna('color', inplace=True)
base_data['color'] = base_data['color'].astype(str)

#Usunięcie colorless i zastąpienie go Colorless
base_data['color'] = base_data['color'].str.strip().str.lower()
base_data['color'].replace('colorless', 'Colorless', inplace=True)

base_data['cut'].fillna('cut', inplace=True)
base_data['cut'] = base_data['cut'].astype(str)

base_data['x dimension'] = pd.to_numeric(base_data['x dimension'], errors='coerce')
mean_value = pd.to_numeric(base_data['x dimension'], errors='coerce').mean(skipna=True)
base_data['x dimension'].fillna(mean_value, inplace=True)
base_data['x dimension'] = base_data['x dimension'].astype(int)

base_data['y dimension'] = pd.to_numeric(base_data['y dimension'], errors='coerce')
mean_value = pd.to_numeric(base_data['y dimension'], errors='coerce').mean(skipna=True)
base_data['y dimension'].fillna(mean_value, inplace=True)
base_data['y dimension'] = base_data['y dimension'].astype(int)

base_data['z dimension'] = pd.to_numeric(base_data['z dimension'], errors='coerce')
mean_value = pd.to_numeric(base_data['z dimension'], errors='coerce').mean(skipna=True)
base_data['z dimension'].fillna(mean_value, inplace=True)
base_data['z dimension'] = base_data['table'].astype(int)

base_data['depth'] = pd.to_numeric(base_data['depth'], errors='coerce')
mean_value = pd.to_numeric(base_data['depth'], errors='coerce').mean(skipna=True)
base_data['depth'].fillna(mean_value, inplace=True)
base_data['depth'] = base_data['depth'].astype(int)


base_data = base_data.drop_duplicates()

base_data.fillna(base_data.mean(), inplace=True)


st.header("Data table")
if st.checkbox("Show/hide data"):
    st.dataframe(base_data)


st.header("Diamonds")

page = st.sidebar.selectbox('Select page',['Price data', 'Rozkład zmiennych numerycznych', 'Liczebność kategorii', 'Fitted Values / Orginalne wartości'])


if page == 'Price data':
    vlist = ['carat', 'x dimension', 'y dimension', 'z dimension', 'depth', 'table']
    column = st.selectbox("Cena vs : ", vlist)
    fig = px.scatter(base_data, x=column, y='price', title=f'Cena vs {column}')
    st.plotly_chart(fig, use_container_width = True)

elif page == 'Liczebność kategorii':
    vlist = ['clarity', 'color', 'cut']
    column = st.selectbox("Liczebność kategorii : ", vlist)
    category_counts = base_data[column].value_counts()

    fig = px.bar(base_data,
                       x=category_counts.index, y=category_counts,
                       title=f'Liczebność kategorii {column}',
                       labels={'x':column, 'y': 'Liczebność'}
                       )
    st.plotly_chart(fig, use_container_width = True)

elif page == 'Fitted Values vs. Original Values of Diamonds':
    base_data = base_data.rename(columns={'x dimension': 'x_dimension', 'y dimension': 'y_dimension', 'z dimension': 'z_dimension'})
    model = smf.ols(formula="price ~ carat + x_dimension + y_dimension + z_dimension + depth + table", data=base_data).fit()
    base_data["fitted"] = model.fittedvalues
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base_data["price"], y=base_data["fitted"], mode='markers', name='Price'))
    fig.add_trace(go.Scatter(x=base_data["price"], y=base_data["price"], mode='lines', name='X vs Y Line'))
    fig.update_layout(title="Fitted Values vs. Original Values of Diamonds",
                  xaxis_title="Original Values of Diamonds",
                  yaxis_title="Fitted Values")
    st.plotly_chart(fig, use_container_width = True)


elif page == 'Rozkład zmiennych numerycznych':
    vlist = ['carat', 'x dimension', 'y dimension', 'z dimension', 'depth', 'table', 'price']
    column = st.selectbox("Rozkład zmiennych numerycznych dla : ", vlist)
    fig = px.histogram(base_data,
                       x=column,
                       title=f'Rozkład zmiennych numerycznych dla {column}'
                       )
    st.plotly_chart(fig, use_container_width = True)    