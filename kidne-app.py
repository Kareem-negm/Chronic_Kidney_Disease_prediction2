
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import webbrowser
from sklearn.model_selection import train_test_split
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier



st.write("""
# Kidney disease prediction application
 Detect if someone has Kidney Disease using Artificial intelligence !
""")


#Get the data
df = pd.read_csv("C:/Users/negmk/Desktop/Chronic_KIdney_Disease_prediction/Chronic_KIdney_Disease_data.csv")
#Show the data as a table (you can also use st.write(df))

st.image("https://www.healtheuropa.eu/wp-content/uploads/2018/04/iStock-650717510-696x392.jpg")

st.subheader('Data Information:')

st.dataframe(df)
#Get statistics on the data
st.write(df.describe())
# Show the data as a chart.
st.set_option('deprecation.showPyplotGlobalUse', False)
f,ax = plt.subplots(figsize=(20, 20))
st.write(sns.heatmap(df.corr(),annot=True))
st.pyplot()

#Split the data into independent 'X' and dependent 'Y' variables
array = df.values
y=array[:,-1]
x_data=array[:,:-1]

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

# Split the dataset into 75% Training set and 25% Testing set
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 33,shuffle=True)


#Get the feature input from the user
def get_user_input():
    age = st.sidebar.slider('age in years', 1, 90, 29)
    bp = st.sidebar.slider("Blood Pressure(numerical) bp in mm/Hg)",50,180,100)
    al = st.sidebar.selectbox('Albumin(nominal)al - (0,1,2,3,4,5)',(0,1,2,3,4,5))
    su = st.sidebar.selectbox(' Sugar(nominal) su - (0,1,2,3,4,5) ',(0,1,2,3,4,5))
    rbc= st.sidebar.selectbox('Red Blood Cells(nominal) rbc - (normal=0,abnormal=1)',(0,1))
    pc = st.sidebar.selectbox("Pus Cell (nominal)pc - (normal=0,abnormal=1)",(0,1))
    pcc = st.sidebar.selectbox('Pus Cell clumps(nominal)pcc - (notpresent =0,present =1)',(0,1))
    ba = st.sidebar.selectbox('Bacteria(nominal) ba - (notpresent =0,present =1)', (0, 1))
    bgr = st.sidebar.slider('Blood Glucose Random(numerical) bgr in mgs/d',22,490,200)
    bu  = st.sidebar.slider('Blood Urea(numerical) bu in mgs/dl',1,391,200)
    sc  = st.sidebar.slider('Serum Creatinine(numerical) sc in mgs/dl',0,76,40)
    pot  = st.sidebar.slider('Potassium(numerical) pot in mEq/L',2,47,30)
    wc = st.sidebar.slider('White Blood Cell Count(numerical) wc in cells/cumm',2200,26400,3300)
    htn = st.sidebar.selectbox('Hypertension(nominal) htn - (yes=1,no=0)',(0,1))
    dm  = st.sidebar.selectbox("Diabetes Mellitus(nominal) dm - (yes=1,no=0)",(0,1))
    cad  = st.sidebar.selectbox('Coronary Artery Disease(nominal) cad - (good=1,poor=0)',(0,1))
    pe = st.sidebar.selectbox('Pedal Edema(nominal) pe - (yes=1,no=0)',(0,1))
    ane  = st.sidebar.selectbox('Anemia(nominal)ane - (yes=1,no=0)',(0,1))
   
    user_data = {'age ': age ,
              'bp ': bp ,
                 'al ': al ,
                 'su ': su ,
              'rbc ': rbc ,
              'pc ': pc ,
                 'pcc ': pcc ,
                 'ba  ': ba  ,
                 'bgr  ': bgr  ,
                 'bu  ': bu  ,
                 'sc  ': sc  ,
                 'pot ': pot ,
                 'wc  ': wc  ,
                 'htn  ': htn  ,
                 'dm  ': dm  ,
                 'cad  ': cad  ,
                 'pe  ': pe  ,
                 'ane  ': ane  ,
                 }
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()
st.subheader('User Input :')
st.write(user_input)


load_clf = pickle.load(open("C:/Users/negmk/Desktop/Chronic_KIdney_Disease_prediction/kidney (1).pkl", 'rb'))
    
predict_button = st.button(label='Predict')

if predict_button:
    prediction = load_clf.predict(user_input)
    st.subheader('Classification: ')
    st.write(prediction)
    
    st.subheader('predicted probabilities: ')
    prediction_proba = load_clf.predict_proba(user_input)
    st.write(prediction_proba)

    if prediction==0:
    
        st.subheader('you dont have Kidney disease , Enjoy and preserve your life')
    else:
        st.subheader('you have Kidney disease , please Click on the next button to go to the tips page and go to the doctor as soon as possible')    
        url = 'https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/managing'

        if st.button('Open browser'):
            webbrowser.open_new_tab(url)
 
    