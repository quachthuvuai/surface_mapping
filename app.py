import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import seaborn as sns
from scipy.interpolate import griddata
import requests  # pip install requests
from streamlit_lottie import st_lottie  # pip install streamlit-lottie



# Page setting
st.set_page_config(layout="wide")

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Wellcome to our App',
                          
                          ['Import and Process data',
                          'Surface Mapping',
                          'Thank You'],
                          icons=['book', 'key', 'people'],
                          default_index=0)
    
with st.sidebar:
    st.title(':arrow_up: Data uploading')
    # st.write('If there is no data uploaded, the default dataset will be used!')
    uploaded_file = st.file_uploader('Upload ASCII (.txt, .dat) data here:')
    path='data/map1.txt'

    # if uploaded_file is None:
    #     df = pd.read_csv(path, delimiter = "\s+", skiprows=20, header=None)
    #     df=df.drop([3, 4], axis=1)
    #     df.columns=['X', 'Y', 'Z']    

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, delimiter = "\s+", skiprows=20, header=None)
        df=df.drop([3, 4], axis=1)
        df.columns=['X (m)', 'Y (m)', 'Z (m)']        


# Information about us
    st.sidebar.title(":two_men_holding_hands: About us")
with st.sidebar.expander("Meet us"):
    st.info(
        """
        This web [app](https://avocadostreamlitapp.herokuapp.com/) developed and maintained by Vu Quach.         
    """
    )

with st.sidebar:
    st.header(":mailbox: Send us your feedback!")


    contact_form = """
    <form action="https://formsubmit.co/quachthuvu.ai@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here"></textarea>
        <button type="submit">Send</button>
    </form>
    """

    st.markdown(contact_form, unsafe_allow_html=True)

    # Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


    local_css("style/style.css")


#=====================================================================================================================================================
if (selected == 'Import and Process data'):
     

    st.header(':bar_chart: Summary of dataset')  

    if st.button("Process your data to workable format"):
        st.info('Your data has been imported and process to workable format. Now you can start to work on your project!')

    # checkboxes
    view_dataframe=st.checkbox('Show Dataset')
    if view_dataframe:
        st.write('Dataset:')
        st.dataframe(df.head(5))
        
    view_statistic=st.checkbox('Show statistic')
    if view_statistic:            
        st.write('Statistics of dataset')
        st.dataframe(df.describe())

    # if uploaded_file is not None:
    #     df = pd.read_csv(uploaded_file, delimiter = "\s+", skiprows=20, header=None)
    #     df=df.drop([3, 4], axis=1)
    #     df.columns=['X (m)', 'Y (m)', 'Z (m)']    

    #     if st.button("Show dataframe and statistics"):
    #         st.info('Dataframe:')
    #         st.dataframe(df.head(5))
    #         st.info('Statistics')
    #         st.dataframe(df.describe())

    X1 = df.iloc[:, 0] 
    Y1 = df.iloc[:, 1] 
    Z1 = df.iloc[:, 2] 

    # 2D dataset
    fig = plt.figure()
    plt.scatter(X1,Y1) #c=color
    # checkboxes
    view_2d_point=st.checkbox('View 2D points')
    if view_2d_point:
        st.write('View 2D points')
        st.pyplot(fig)

    # 3D scatterplot
    fig0 = plt.figure() 
    ax0 = fig0.add_subplot(111, projection='3d') 
    surf=ax0.scatter(X1, Y1, Z1, c=Z1) 
    ax0.set_title('3D view of points')
    fig0.colorbar(surf, ax = ax0, shrink = 0.5, aspect = 5, orientation="vertical", anchor=(0.5,1))
    ax0.xaxis.set_major_locator(MaxNLocator(5))
    ax0.yaxis.set_major_locator(MaxNLocator(6))
    ax0.zaxis.set_major_locator(MaxNLocator(5))
    ax0.set_title('3D view of triangle surface')
    ax0.set_xlabel('X-axis', fontweight ='bold')
    ax0.set_ylabel('Y-axis', fontweight ='bold')
    ax0.set_zlabel('Z-axis', fontweight ='bold')
    # checkboxes
    view_3d_point=st.checkbox('View 3D points')
    if view_3d_point:
        st.write('3D point of raw data')
        st.pyplot(fig0)

 #=====================================================================================================================================================
elif (selected == 'Surface Mapping'):

    st.header(':pushpin: Surface Mapping')  

    nx = 48 # x dimension
    ny = 22 # y dimension
    x = df['X (m)'].to_numpy()
    y = df['Y (m)'].to_numpy()
    z = df['Z (m)'].to_numpy()

    xv = np.linspace(np.min(x), np.max(x), 1056)
    yv = np.linspace(np.min(y), np.max(y), 1056)

    X, Y = np.meshgrid(xv, yv)

    Z = griddata((x,y),z,(X,Y),method='linear')

    
    # 3D surface plot
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7,5))
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.jet,
                        linewidth=0, antialiased=False)
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    ax1.set_title('3D view of 3D surface')
    fig1.colorbar(surf, ax = ax1, shrink = 0.5, aspect = 5, orientation="vertical", anchor=(1,1))

    ax1.xaxis.set_major_locator(MaxNLocator(5))
    ax1.yaxis.set_major_locator(MaxNLocator(6))
    ax1.zaxis.set_major_locator(MaxNLocator(5))
    # checkboxes
    view_3d_surface=st.checkbox('3D Surface')
    if view_3d_surface:
        st.write('3D surface of depth map')
        st.pyplot(fig1)

    # 3D wireframe plot 
    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    ax2.plot_wireframe(X, Y, Z, rstride=50, cstride=50)
    ax2.set(xticklabels=[],
        yticklabels=[],
        zticklabels=[])
    # checkboxes
    view_3d_wireframe=st.checkbox('3D Wireframe')
    if view_3d_wireframe:
        st.write('3D wireframe of depth map')
        st.pyplot(fig2)

    # 2D surface and contours   
    # # checkboxes
    view_2d=st.checkbox('2D View')   
    increment = st.slider('Set contour increment', 5, 100, 20, step=5)
    no_countours=round((np.max(Z)-np.min(Z))/increment)    
    levels = np.linspace(np.min(Z), np.max(Z), no_countours)
    # left fig
    fig3, (ax1,ax2) = plt.subplots(1,2, figsize=(12,4))
    cp1=ax1.contour(X, Y, Z,colors='b', levels=levels)
    ax1.clabel(cp1, inline=True, fontsize=10)
    ax1.set_title('Contour Plot')
    ax1.set_xlabel('x (cm)')
    ax1.set_ylabel('y (cm)')
    # right fig
    cp2 = ax2.pcolormesh(X, Y, Z)
    cp3=ax2.contour(X, Y, Z,colors='b', levels=levels)
    ax2.clabel(cp3, inline=True, fontsize=12)
    ax2.set_title("2D surface plot")
    ax2.set_xlabel("x (cm)")
    ax2.set_ylabel("y (cm)")   
    fig3.colorbar(cp2)
    
    if view_2d:
        st.write('2D view of Layang depth map')
        st.pyplot(fig3)

#=====================================================================================================================================================
elif (selected == 'Thank You'):
    st.balloons()
    st.header('THANK YOU FOR YOUR VISITING OUR APP') 

    
    # def load_lottiefile(filepath: str):
    #     with open(filepath, "r") as f:
    #         return json.load(f)


    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    

    #lottie_coding = load_lottiefile("images/125768-mobile-app.json")  
    lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_totrpclr.json")

    st_lottie(
        lottie_hello,
        speed=1,
        reverse=False,
        loop=True,
        quality="low", # medium ; high
        # renderer="svg", # canvas
        height=None,
        width=None,
        key=None,
    )


        
   
    
    


        


# data=data.sort_values("Date")
# data.drop(['Unnamed: 0'], axis=1, inplace=True)


# avocado_stats = data.groupby('type')['AveragePrice', 'Total Volume', 'Total Bags'].mean()


# data['week']=data.Date.dt.isocalendar().week # https://github.com/pandas-dev/pandas/issues/39142
# data['month']=pd.DatetimeIndex(data['Date']).month

# def convert_month(month):
#   if month==3 or month==4 or month==5:
#     return 0
#   elif month==6 or month==7 or month==8:
#     return 1
#   elif month==9 or month==10 or month==11:
#     return 2
#   else:
#     return 3

# data['season']=data['month'].apply(lambda x: convert_month(x))

# week_price=data.groupby('week').mean()
# month_price=data.groupby('month').mean()
# season_price=data.groupby('season').mean()
# year_price=data.groupby('year').mean()


# #Regression RandomForest Model
# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()

# data['type_encoded']=le.fit_transform(data['type'])

# data_ohe=pd.get_dummies(data=data, columns=['region'])

# X=data_ohe.drop(['Date', 'AveragePrice', '4046', '4225', '4770',
#         'Small Bags', 'Large Bags', 'XLarge Bags', 'type'], axis=1)
# y=data_ohe['AveragePrice']


# X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=0)


# pipe_RF=Pipeline([('scaler', StandardScaler()), 
#                   ('rf', RandomForestRegressor())])

# pipe_RF.fit(X_train, y_train)

# y_pred_RF=pipe_RF.predict(X_test)




# # Reads test data
# organic_data = pd.read_csv('organic_test.csv')
# organic_test=organic_data.set_index(pd.DatetimeIndex(organic_data['Date'])).drop('Date', axis=1)
# # Reads test data
# conventional_data = pd.read_csv('organic_test.csv')
# conventional_test=conventional_data.set_index(pd.DatetimeIndex(conventional_data['Date'])).drop('Date', axis=1)


# # Reads in saved arima model
# organic_model = pickle.load(open('arima_model_organic.pkl', 'rb'))
# conventional_model = pickle.load(open('arima_model_conventional.pkl', 'rb'))


# # Reads in saved prophet model
# df_pf_or = pd.read_csv('df_pf_or.csv', parse_dates=['ds'])
# df_pf_or=df_pf_or.drop('Unnamed: 0', axis=1)
# df_pf_or.y = df_pf_or.y.astype(float)

# pf_model_or = Prophet(yearly_seasonality=True, \
#             daily_seasonality=False, weekly_seasonality=False)

# pf_model_organic=pf_model_or.fit(df_pf_or)

# # Reads in saved prophet files
# df_pf_con = pd.read_csv('df_pf_con.csv', parse_dates=['ds'])
# df_pf_con=df_pf_con.drop('Unnamed: 0', axis=1)
# df_pf_con.y = df_pf_con.y.astype(float)

# pf_model_con = Prophet(yearly_seasonality=True, \
#             daily_seasonality=False, weekly_seasonality=False)

# pf_model_conventional=pf_model_con.fit(df_pf_con)

# #=====================================================================================================================================================

# if (selected == 'Import and Process data'):

#     st.markdown("""
#     # :books: Business Objective
#     **Hass**, a company based in Mexico, specializes in producing a variety of avocados for selling in U.S.A. 
#     They have been very successful over recent years and want to expand their business. 
#     Thus they want build a reasonable model to predict the average price of avocado “Hass” in the U.S.A to consider the expansion of existing avocado farms in other regions.

#     There are two types of avocados (**conventional** and **organic**) in the dataset and several different regions. 
#     This allows us to do analysis for either conventional or organic avocados in different regions and/or the entire United States.

#     **There are 2 different approaches to solve this business objective:**
#     - **First approach:** create a **regression model** using supervised machine learning algorithms such as **Linear Regression, Random Forest, XGB Regressor** so on to predict average price of avocado in the USA.
#     - **Second approach:** build a **predictive model** based on supervised time-series machine learning algorithms like **Arima, Prophet, HoltWinters** to predict average price of a particular avocado (organic or conventional) over time for a specific region in the USA.
#     """)

#     from PIL import Image 
#     img = Image.open("images/avocado_1.jpg")
#     st.image(img,width=700,caption='Streamlit Images')

#     st.write("""
#     # :chart_with_upwards_trend: Avocado price Forecasting
#     This app used **avocado.csv** dataset as train and test data. 
#     """)
#     st.markdown('''
#     This is a dashboard showing the **average prices** of different types of :avocado:  
#     Data source: [Kaggle](https://www.kaggle.com/datasets/timmate/avocado-prices-2020)
#     ''')

#     st.info(" **OBJECTIVE:** To predict / forecast the average price of Avocado based on time series data using ***regression model***, ***ARIMA model***, ***PROPHET model*** ")


#     st.header(':bar_chart: Summary statistics')

#     st.dataframe(avocado_stats)

#     st.header(':triangular_flag_on_post: Avocado Price and Total volume by geographies')

#     b1, b2, b3, b4 = st.columns(4)
#     region=b1.selectbox("Region", np.sort(data.region.unique()))
#     avocado_type=b2.selectbox("Avocado Type", data.type.unique())

#     start_date=b3.date_input("Start Date",
#                                     data.Date.min().date(),
#                                     min_value=data.Date.min().date(),
#                                     max_value=data.Date.max().date(),
#                                     )

#     end_date=b4.date_input("End Date",
#                                     data.Date.max().date(),
#                                     min_value=data.Date.min().date(),
#                                     max_value=data.Date.max().date(),
#                                     )

#     mask = ((data.region == region) &
#             (data.type == avocado_type) &
#             (data.Date >= pd.Timestamp(start_date)) &
#             (data.Date <= pd.Timestamp(end_date)))
#     filtered_data = data.loc[mask, :] 



#     with st.form('line_chart'):
#         filtered_avocado = filtered_data
#         submitted = st.form_submit_button('Submit')
#         check_box=st.checkbox("Included Sale Volume")

#         if submitted:
            
#             price_fig = px.line(filtered_avocado,
#                             x='Date', y='AveragePrice',
#                             color='type',
#                             title=f'Avocado Price in {avocado_type}')
#             st.plotly_chart(price_fig)
            
#             # Show sale volume
#             if check_box:	 
#                 volume_fig = px.line(filtered_avocado,
#                                 x='Date', y='Total Volume',
#                                 color='type',
#                                 title=f'Avocado Sale Volume in {avocado_type}')           
#                 st.plotly_chart(volume_fig)


# #=====================================================================================================================================================

# elif (selected == 'Surface Mapping'):
#     st.header(':pushpin: Data Exploration')
#     st.subheader(':memo: Avocado data summary')
#     st.dataframe(data.describe())

#     st.subheader(':chart_with_downwards_trend: Seasonality analysis')
#     Byweekly = st.checkbox('Weekly')
#     if Byweekly:
#         st.success('🌎 **Seasonality by weekly**')
#         fig_weekly=plt.figure(figsize=(10,6))
#         plt.plot(week_price.index, week_price['AveragePrice'])
#         st.pyplot(fig_weekly)
#     Bymonthly = st.checkbox('Monthly')
#     if Bymonthly:
#         st.success('🌎 **Seasonality by monthly**')
#         fig_monthly=plt.figure(figsize=(10,6))
#         plt.plot(month_price.index, month_price['AveragePrice'])
#         st.pyplot(fig_monthly)
#     BySeason = st.checkbox('BySeason')
#     if BySeason:
#         st.success('🌎 **Seasonality by season**')
#         fig_season=plt.figure(figsize=(10,6))
#         plt.plot(season_price.index, season_price['AveragePrice'])
#         st.pyplot(fig_season)
#     Byyearly = st.checkbox('Yearly')
#     if Byyearly:
#         st.success('🌎 **Seasonality by yearly**')
#         fig_yearly=plt.figure(figsize=(10,6))
#         plt.plot(year_price.index, year_price['AveragePrice'])
#         st.pyplot(fig_yearly)
    
#     st.subheader(':chart: Price analysis')
#     columns = st.multiselect(label='Please select type of avocado to check the average price change by region', options=data.type.unique())
#     if st.button("Generate Price Plot"):

#         if columns==['organic']:
            
#             st.write("classify by region, filter type=='organic' ")
#             sns.set(style='whitegrid')
#             fig1=plt.figure(figsize=(25,5))
#             sns.boxplot(data=data[data['type']=='organic'], x='region', y='AveragePrice')
#             plt.xticks(rotation=90)
#             st.pyplot(fig1)

#         elif columns==['conventional']:

#             st.write("classify by region, filter type=='conventional' ")
#             sns.set(style='whitegrid')
#             fig2=plt.figure(figsize=(25,5))
#             sns.boxplot(data=data[data['type']=='conventional'], x='region', y='AveragePrice')
#             plt.xticks(rotation=90)
#             st.pyplot(fig2)  

#         elif columns==['organic', 'conventional']:
            
#             st.write("classify by region, filter type==['organic'] ")
#             fig3=plt.figure(figsize=(25,5))
#             sns.boxplot(data=data[data['type']=='organic'], x='region', y='AveragePrice')
#             plt.xticks(rotation=90)
#             st.pyplot(fig3)

#             st.write("classify by region, filter type==['conventinal'] ")
#             fig4=plt.figure(figsize=(25,5))
#             sns.boxplot(data=data[data['type']=='conventional'], x='region', y='AveragePrice')
#             plt.xticks(rotation=90)
#             st.pyplot(fig4)
#         else:
#             st.write('please select again!')
#     st.markdown('''
#         - price change by region
#         - some region has higher price
#         - some region has lower price                   
#         ''')

#     st.subheader(':chart: Sale volume analysis')
#     columns_1 = st.multiselect(label='Please select type of avocado to check the sale volume change by region', options=data.type.unique())
#     if st.button("Generate Volume Plot"):

#         if columns_1==['organic']:
            
#             st.write("classify by region, filter type=='organic' ")
#             sns.set(style='whitegrid')
#             fig_1=plt.figure(figsize=(25,5))
#             sns.boxplot(data=data[(data['type']=='organic') & (data['region']!='TotalUS')], x='region', y='Total Volume')
#             plt.xticks(rotation=90)
#             st.pyplot(fig_1)

#         elif columns_1==['conventional']:

#             st.write("classify by region, filter type=='conventional' ")
#             sns.set(style='whitegrid')
#             fig_2=plt.figure(figsize=(25,5))
#             sns.boxplot(data=data[(data['type']=='conventional') & (data['region']!='TotalUS')], x='region', y='Total Volume')
#             plt.xticks(rotation=90)
#             st.pyplot(fig_2)  

#         elif columns_1==['organic', 'conventional']:
            
#             st.write("classify by region, filter type==['organic'] ")
#             fig_3=plt.figure(figsize=(25,5))
#             sns.boxplot(data=data[(data['type']=='organic') & (data['region']!='TotalUS')], x='region', y='Total Volume')
#             plt.xticks(rotation=90)
#             st.pyplot(fig_3)

#             st.write("classify by region, filter type==['conventinal'] ")
#             fig_4=plt.figure(figsize=(25,5))
#             sns.boxplot(data=data[(data['type']=='conventional') & (data['region']!='TotalUS')], x='region', y='Total Volume')
#             plt.xticks(rotation=90)
#             st.pyplot(fig_4)
#         else:
#             st.write('please select again!')

#     st.markdown('''
#         Grouped region from multiple states: show high sale volume

#         - southest region
#         - northest region
#         - southcentral region
#         - midsouth region
#         - west region
                
#         LosAngles city is belong to California state. California is the Largest avocado Consumer in the US
            
#         ''')
#     text=" In this exercise we will focus on **California** region because California has higher price and higher sale volume"
#     new_title = ':point_right:' + '<p style="font-family:sans-serif; color:Blue; font-size: 30px;">' + text + '</p>'
#     st.markdown(new_title, unsafe_allow_html=True)

    
# #=====================================================================================================================================================    

# elif (selected == 'Thank You'): 
#     st.balloons()
#     st.header('THANK YOU FOR YOUR LISTENING') 


#     # def load_lottiefile(filepath: str):
#     #     with open(filepath, "r") as f:
#     #         return json.load(f)


#     def load_lottieurl(url: str):
#         r = requests.get(url)
#         if r.status_code != 200:
#             return None
#         return r.json()
    

#     #lottie_coding = load_lottiefile("images/125768-mobile-app.json")  
#     lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_totrpclr.json")

#     st_lottie(
#         lottie_hello,
#         speed=1,
#         reverse=False,
#         loop=True,
#         quality="low", # medium ; high
#         # renderer="svg", # canvas
#         height=None,
#         width=None,
#         key=None,)
#