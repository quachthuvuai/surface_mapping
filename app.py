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

# default dataset
df = pd.read_csv('data/map1.txt', delimiter = "\s+", skiprows=20, header=None)
df=df.drop([3, 4], axis=1)
df.columns=['X (m)', 'Y (m)', 'Z (m)'] 
X1 = df.iloc[:, 0] 
Y1 = df.iloc[:, 1] 
Z1 = df.iloc[:, 2] 

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
