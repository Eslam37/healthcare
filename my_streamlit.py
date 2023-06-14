import streamlit as st
import pandas as pd
import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import hydralit_components as hc 
import streamlit as st
import base64

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

st.set_page_config(
    page_title="Drinking Water",
    page_icon= "water.jpg",
    layout='wide'
)


#Creating Navigation bar
menu_data = [
    {'label': "Problem Analysis", 'icon': '🔍'},
    {'label': 'Predictive Model', 'icon': '🧪'},
    {'label': 'Recommendations', 'icon': '📋'}
]
menu_id = hc.nav_bar(menu_definition=menu_data, sticky_mode='sticky', override_theme={ 'menu_background': '#3A9BCD', 'option_active': 'white'}
)

if menu_id =="Problem Analysis":
  st.header("Access to Clean Water")
  st.markdown("### Unsafe water remains a significant global issue, causing more than <span style='color:red'>**1.2 million deaths**</span> per year",unsafe_allow_html=True)
  col1,col2 = st.columns([6,3])

  # Display the animated .mp4 video in the first column
  col1, col2 = st.columns([3,1])
  col1.video("Causes and effects of water pollution - Sustainability _ ACCIONA.mp4")
  # col2.markdown('<img src="wash-hand.gif"/>', unsafe_allow_html=True)
  
  """### gif from local file"""
  file_ = open("wash-hand.gif", "rb")
  contents = file_.read()
  data_url = base64.b64encode(contents).decode("utf-8")
  file_.close()

  col2.markdown(
      f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" width="300" height="300">',
      unsafe_allow_html=True,
  )

  
  st.markdown("### Total Number of Deaths for Each Risk Factor",unsafe_allow_html=True)
  df = pd.read_csv("number-of-deaths-by-risk-factor.csv")
  df.head()
  df.sort_values(by = "deaths",ascending=True)
  labels = df['risk_factor']
  values = df['deaths']
  colors = ['#b3b7bd',] * labels.nunique()
  colors[0] = '#3a9bcd'

  fig = go.Figure([
      go.Bar(y=labels, x=values, orientation ='h', text=values, name='Deaths', marker_color=colors
  ),
  ])
  fig.update_layout(plot_bgcolor = 'rgba(0,0,0,0)', title="deaths".title(), yaxis={'categoryorder':'total ascending'})

  col1, col2 = st.columns(2)
  col1.plotly_chart(fig)
  df = pd.read_csv("improved_santation_facilities.csv")
  regions = df['Region']
  c1 = df['1990']
  c2 = df['2015']
  fig = go.Figure([
      go.Bar(y=regions, x=c1, orientation ='h', text=c1, name='1990'),
      go.Bar(y=regions, x=c2, orientation ='h', text=c2, name='2015'),
  ])
  fig.update_layout(barmode='group', plot_bgcolor = 'rgba(0,0,0,0)', title="population percentage with imporved santiation facilities per region".title())
  col2.plotly_chart(fig)

  st.markdown("### Unequal Access: Water and Sanitation in sub-Saharan Africa VS Europe and US",unsafe_allow_html=True)
  theme_bad = {'bgcolor':'#FFF0F0','content_color':'darkred','progress_color':'darkred'}
  theme_neutral = {'bgcolor': '#3A9BCD','title_color': 'darkblue','content_color': 'darkblue','icon_color': 'darkblue', 'icon': 'fa fa-question-circle'}
  theme_good = {'bgcolor': '#EFF8F7','title_color': 'green','content_color': 'green','icon_color': 'green', 'icon': 'fa fa-check-circle'}

  cc = st.columns(4)

  with cc[0]:
    # can just use 'good', 'bad', 'neutral' sentiment to auto color the card
    hc.info_card(title='Access to drinking water in sub-Saharan Africa', content='39% With No Access',bar_value=39,theme_override=theme_bad)

  with cc[1]:
    hc.info_card(title='Access to sanitation in sub-Saharan Africa', content='69% With No Access',bar_value=69,theme_override=theme_bad)

  with cc[2]:
    hc.info_card(title='Access to drinking water in Europian Countries', content='97.81% With Access', sentiment='good',bar_value=98)

  with cc[3]:
  #customise the the theming for a neutral content
    hc.info_card(title='Access to drinking water in the United States', content='95% With Access', sentiment='good',bar_value=95)


  st.markdown("### Total Deaths per Country due to unsafe water drinking from 1990 till 2019", unsafe_allow_html=True)
  df = pd.read_csv(r'C:\Users\eslam\OneDrive\Desktop\MSBA\MSBA 350E\Individual Project\Streamlit\share-deaths-unsafe-water.csv')
  col1, col2 = st.columns([3,1])
  # Create an animated choropleth map
  fig = px.choropleth(df, locations='country', locationmode='country names',
                      color='deaths', hover_name='country', animation_frame='year',
                      title='Total Deaths by Country (Animated Map)',
                      color_continuous_scale='Reds', range_color=(df['deaths'].max(), 0),
                      labels={'deaths': 'Total Deaths'})

  # Set the range of the color scale
  fig.update_layout(coloraxis_colorbar=dict(title='Deaths Percent'))

  # Adjust the size of the plot
  fig.update_layout(height=600, width=1100)

  # Increase the animation speed
  fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 300

  # Show the animated plot
  col1.plotly_chart(fig)
  col2.image("droppen.gif",use_column_width=True)

  st.markdown("### Total Deaths per Country due to unsafe water drinking from 1990 till 2019", unsafe_allow_html=True)
  html_code = '''<iframe src="https://ourworldindata.org/grapher/access-drinking-water-stacked"  style="width: 100%; height: 800px; border: 0px none;"></iframe>'''
  st.components.v1.html(html_code)


# Life expectancy page
if menu_id == "Predictive Model":
    df = pd.read_csv('water_potability.csv')
    st.header("Let us predict the water potability.... fill in the following information first!")

    # Create the data for the table
    table = [
        ["pH value", "parameter to evaluating the acid–base balance of water"],
        ["Hardness", "the capacity of water to precipitate soap caused by Calcium and Magnesium"],
        ["Solids", "Total dissolved solids - TDS where high TDS means water is highly mineralized"],
        ["Chloramines", "Chloramines are most commonly formed when ammonia is added to chlorine to treat drinking water"],
        ["Sulfate", "Sulfates are naturally occurring substances that are found in minerals, soil, and rocks"],
        ["Conductivity", "the amount of dissolved solids in water determines the electrical conductivity"],
        ["Organic_carbon", "TOC is a measure of the total amount of carbon in organic compounds in pure water."],
        ["Trihalomethanes", "THMs are chemicals which may be found in water treated with chlorine"],
        ["Turbidity", "It depends on the quantity of solid matter present in the suspended state"],
    ]

    # Create a DataFrame from the data
    d = pd.DataFrame(table, columns=["Feature", "Description"])

    # Add CSS classes for styling
    styles = [
        {"selector": "table", "props": [("border-collapse", "collapse")]},
        {"selector": "th, td", "props": [("border", "1px solid #ddd"), ("padding", "8px")]},
        {"selector": "th", "props": [("background-color", "#f2f2f2")]},
    ]

    # Display the styled table
    style_tags = "".join(f"{style['selector']}{{{';'.join([f'{prop[0]}:{prop[1]}' for prop in style['props']])}}}" for style in styles)
    st.markdown(f'<style>{style_tags}</style>', unsafe_allow_html=True)
    st.table(d.style.hide_index())

    if st.checkbox('Show data sample'):
        st.subheader('Raw data')
        st.write(df)

    col1a, col1b, col1c = st.columns([3, 3, 3])
    with col1a:
        ## sliders for machine learning prediction
        ph = st.slider(
            "ph", min_value=0, max_value=14, step=1, value=1
        )
        df = df[df["ph"] < ph]

        hardness = st.slider(
            "Hardness", min_value=47, max_value=323, step=10, value=60
        )
        df = df[df["Hardness"] < hardness]

        solids = st.number_input(
            "Solids", min_value=0, max_value=10000, step=100, value=0
        )
        df = df[df["Solids"] < solids]

    with col1b:
        ## sliders for machine learning prediction
        chloramines = st.slider(
            "Chloramines", min_value=0, max_value=14, step=1, value=1
        )
        df = df[df["Chloramines"] < chloramines]

        sulfate = st.slider(
            "Sulfate", min_value=100, max_value=500, step=10, value=60
        )
        df = df[df["Sulfate"] < sulfate]

        conductivity = st.number_input(
            "Conductivity", min_value=0, max_value=10000, step=100, value=0
        )
        df = df[df["Conductivity"] < conductivity]

    with col1c:
        ## sliders for machine learning prediction
        carbon = st.slider(
            "Organic Carbon", min_value=0, max_value=30, step=1, value=1
        )
        df = df[df["Organic_carbon"] < carbon]

        trihalomethanes = st.slider(
            "Trihalomethanes", min_value=0, max_value=150, step=10, value=60
        )
        df = df[df["Trihalomethanes"] < trihalomethanes]

        turbidity = st.number_input(
            "Turbidity",min_value=0, max_value=10000, step=100, value=0
        )
        df = df[df["Turbidity"] < turbidity]

    data = pd.read_csv('water_potability.csv')

    X = data.drop("Potability", axis=1)
    y = data["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC())
    ])

    pipeline.fit(X_train, y_train)

    new_data = pd.DataFrame({
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }, index=[0])
    prediction = pipeline.predict(new_data)

    if st.button ("Check Water Potability"):
      st.subheader(prediction)
      st.markdown("""Please note that the factors used to determine Water potability are limited to the Country Scale Factors. and limited to the data which the model was trained on!""")






# st.title('MSBA 350E')

# DATE_COLUMN = 'date/time'
# # DATA_URL = ('C:\Users\eslam\OneDrive\Desktop\MSBA\MSBA325\sales.py')

# @st.cache
# def load_data(nrows):
#     data = pd.read_csv(r'C:\Users\eslam\OneDrive\Desktop\MSBA\MSBA 350E\Individual Project\Streamlit\number-of-deaths-by-risk-factor.csv', encoding= 'unicode_escape', nrows=nrows , )
#     # lowercase = lambda x: str(x).lower()
#     # data.rename(lowercase, axis='columns', inplace=True)
#     # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# data_load_state = st.text('Loading data...')
# data = load_data(10000)
# data_load_state.text("Done! (using st.cache)")

# import streamlit as st

# # Create two columns
# col1, col2 = st.columns(2)

# # Display the animated .mp4 video in the first column
# col1.video("Causes and effects of water pollution - Sustainability _ ACCIONA.mp4")

# # Display the animated .gif image in the second column
# col2.image("droppen.gif", use_column_width=True)



# st.text('This data is about sales of a company which shows the id of products, quantity sold, prices, sales, county from where people are purchasing and we will analyze the given data to see how the company is doing.')

# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)


# st.subheader("Data Visualization 1")
# st.text('This visualization show the different products in the company and the amount of sales for each product over three years and the animations show how the sales of the product are changeing over time')

# df = pd.read_csv("number-of-deaths-by-risk-factor.csv", encoding='unicode_escape')

# colors = ['gray'] * len(df['risk_factor'])
# colors[df[df['risk_factor'] == 'Unsafe water source'].index[0]] = 'red'

# fig = go.Figure()
# fig.add_trace(go.Bar(
#     y=df['risk_factor'],
#     x=df['deaths'],
#     orientation='h',
#     text=df['deaths'],
#     textposition='outside',
#     textfont={'size': 14},  # Set the font size for the text labels
#     marker_color=colors,
# ))

# fig.update_layout(
#     title='Number of Deaths by Risk Factor',
#     xaxis_title='Number of Deaths',
#     yaxis_title='Risk Factor',
#     xaxis_showgrid=False,  # Remove the x-axis grid
#     yaxis_showgrid=False,  # Remove the y-axis grid
#     plot_bgcolor='rgba(0,0,0,0)',  # Set the plot background color to transparent
#     paper_bgcolor='rgba(0,0,0,0)',  # Set the paper background color to transparent
# )

# fig.update_xaxes(showticklabels=False)
# st.plotly_chart(fig)



# st.subheader("Data Visualization 2")
# st.text('This visualization show the different countries where customers are purchasing products from and the amount of sales we have from each country')

# df = pd.read_csv(r'C:\Users\eslam\OneDrive\Desktop\MSBA\MSBA 350E\Individual Project\Streamlit\share-deaths-unsafe-water.csv')

# # Create an animated choropleth map
# fig = px.choropleth(df, locations='country', locationmode='country names',
#                     color='deaths', hover_name='country', animation_frame='year',
#                     title='Total Deaths by Country (Animated Map)',
#                     color_continuous_scale='Reds', range_color=(df['deaths'].max(), 0),
#                     labels={'deaths': 'Total Deaths'})

# # Set the range of the color scale
# fig.update_layout(coloraxis_colorbar=dict(title='Deaths Percent'))

# # Show the animated plot
# st.plotly_chart(fig)


# import streamlit as st

# html_code = '''
# <iframe src="https://ourworldindata.org/grapher/access-drinking-water-stacked" loading="lazy" style="width: 100%; height: 600px; border: 0px none;"></iframe>
# '''

# st.components.v1.html(html_code)



# import pandas as pd
# import plotly.express as px
# import streamlit as st



