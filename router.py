import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import joblib
import pickle

# global variables
cwd = os.getcwd()
df = pd.read_csv(os.path.join(cwd, 'dataset.csv'))
df_cleaned = pd.read_csv(os.path.join(cwd, 'dataset_cleaned.csv'))

with open("m_pipe.pickle", "rb") as file:
    main_pipe = pickle.load(file)

rf_model = joblib.load(os.path.join(cwd, "models", "rf_model.h5"))
knn_model = joblib.load(os.path.join(cwd, "models", "knn_model.h5"))

# page data
st.set_page_config(
    layout='wide',
    page_title='Laptops prices',
    page_icon='💻'
)

# navigation sidebar
side_bar = st.sidebar.radio(':green[Select page]', ['Dataset overview', 'Descriptive statistics', 'Charts', "Prediction"])


# page1
if side_bar == 'Dataset overview':
    st.write('<h1 style = "text-align: center; color: #00ef6d;">Dataset overview</h1>', unsafe_allow_html=True)
    space1, col, space2 = st.columns([1, 7, 1])
    col.dataframe(df, width=None, height=700)


# page2
elif side_bar == 'Descriptive statistics':
    st.write('<h1 style = "text-align: center; color: #00ef6d;">Descriptive statistics</h1>', unsafe_allow_html=True)
    num_col, cat_col = st.columns([5, 2])
    with num_col:
        st.write('<h3 style = "text-align: 10%; color: #00ef6d;">Numerical columns</h3>', unsafe_allow_html=True)
        st.dataframe(df_cleaned.describe())
    with cat_col:
        st.write('<h3 style = "text-align: 10%; color: #00ef6d;">Categorical columns</h3>', unsafe_allow_html=True)
        st.dataframe(df_cleaned.describe(include="object"))


# page 3
elif side_bar == "Charts":
    tab1, tab2, tab3, tab4 = st.tabs(['UniVariate', 'BiVariate', "PolyVariate", "SunBurst"])

## tab1
    with tab1:
        col_name = st.radio(":green[select a variable]", df_cleaned.columns)
        fig1 = px.histogram(df_cleaned, x=col_name, color_discrete_sequence=["#00ef6d"], title=f'{col_name} hist distribution', width=550)
        st.plotly_chart(fig1)
## tab2
    with tab2:
        col1, col2 = st.columns([5, 5])
        with col1:
            st.write('<h3 style = "text-align: 10%; color: #00ef6d;">X-axis variable</h3>', unsafe_allow_html=True)
            col_name_x = st.radio(":green[select a variable]", df_cleaned.columns, key="t2x")
            
        with col2:
            st.write('<h3 style = "text-align: 10%; color: #00ef6d;">Y-axis variable</h3>', unsafe_allow_html=True)
            col_name_y = st.radio(":green[select a variable]", df_cleaned.columns, key="t2y")

        p3_t2_fig1 = px.scatter(df_cleaned, x=col_name_x, y=col_name_y, color="price", width=550, color_continuous_scale="Greens")
        st.plotly_chart(p3_t2_fig1)
## tab3
    with tab3:
        st.write('<h3 style = "text-align: 10%; color: #00ef6d;">Coming soon</h3>', unsafe_allow_html=True)

## tab4
    with tab4:
        col_name_4 = st.radio(":green[select a variable]", df_cleaned.columns.drop(["brand", "price"]), key="t4m")
        p3_t4_fig1 = px.sunburst(df_cleaned, path=['brand', col_name_4], color_discrete_sequence=px.colors.qualitative.Antique)
        st.plotly_chart(p3_t4_fig1)
# page 4
elif side_bar == "Prediction":

    st.write('<h1 style = "text-align: center; color: #00ef6d;">Price Prediction</h1>', unsafe_allow_html=True)

    brand = st.selectbox(':green[Brand]', options=df_cleaned.brand.value_counts().index, placeholder="choose laptop brand")
    ram = st.number_input(':green[RAM]', value=8, step=1)
    hard_disk = st.number_input(':green[Hard disk]', value=1000, step=1)
    screen_size = st.number_input(':green[screen size]', value=14, step=1)


    st.write('<h4 style = "text-align: 10%; color: #00ef6d;">Choose a model</h4>', unsafe_allow_html=True)
    col1, col2 = st.columns([5, 5])
    with col1:
        knn_check = st.checkbox('k-nearest neighbors')
    with col2:
        rf_check = st.checkbox('random forest')


    if st.button('process'):
        # prediction precess
        ## collecting the values from the variable and store them as dataframe
        new_data = pd.DataFrame([np.array([brand, screen_size, hard_disk, ram])], columns=df_cleaned.columns.drop("price"))

        ## converting the datatypes
        new_data["screen_size"] = new_data["screen_size"].astype("float")
        new_data["harddisk"] = new_data["harddisk"].astype("float")
        new_data["ram"] = new_data["ram"].astype("float")

        new_data = main_pipe.transform(new_data)

        ## prediction process
        y_pred = []
        if any([knn_check, rf_check]):
            if knn_check:
                y_pred.append(knn_model.predict(new_data)[0])
            if rf_check:
                y_pred.append(rf_model.predict(new_data)[0])

            st.success(f'expected price: {round(sum(y_pred) / len(y_pred), 2)}$')
        else:
            st.success("select at least one model!")
        

else:
    st.write('<h1 style = "text-align: center; color: #00ef6d;">Error 501: Not Implemented</h1>', unsafe_allow_html=True)