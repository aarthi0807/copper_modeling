import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder

st.set_page_config(layout="wide", page_title="Industrial copper modeling")
st.markdown("<h1 style='text-align: center; color: black;'>Industrial Copper Modeling</h1>", unsafe_allow_html=True)
tab1,tab2 = st.tabs(["Predict selling price", "Predict status"])
with tab1:
    col1,col2 = st.columns(2)
    with col1:
        quantity_tons = st.text_input("Quantity tons (Min:0, Max:1000000000)")
        customer = st.text_input("Customer (Min:12458, Max:30408185)")
        country = st.selectbox("Country",(0,28,25,30,32,38,78,27,77,113,79,26,39,40,84,80,107,89))
        application = st.selectbox("Application",(0,10,41,28,59,15,4,38,56,42,26,27,19,20,66,29,22,40,25,67,79,3,99,2,5,39,69,70,65,58,68))
        thickness = st.text_input("Thickness (Min:0.18 & Max:400)")
        width = st.text_input("Width (Min:1, Max:2990)")
    with col2:
        status = st.selectbox("Status",("Won","Draft","To be approved","Lost","Not lost for AM","Wonderful","Revised","Offered","Offerable"))
        item_type = st.selectbox("Item type",("W","WI","S","Others","PL","IPL","SLAWR"))
        prod_ref = st.selectbox("Product Reference",(1670798778,1668701718,628377,640665,611993,1668701376,164141591,1671863738,1332077137,640405,1693867550,1665572374,1282007633,1668701698,628117,1690738206,628112,640400,1671876026,164336407,164337175,1668701725,1665572032,611728,1721130331,1693867563,611733,1690738219,1722207579,929423819,1665584320,1665584662,1665584642))
        predict = st.button("Predict Selling Price")
        if predict:
            if quantity_tons and customer and thickness and width is not None:
                with open("C:/Users/Raghavendra Kumar JR/Dump/model.pkl", "rb") as file:
                    model = pickle.load(file)
                with open("C:/Users/Raghavendra Kumar JR/Dump/scaler.pkl", "rb") as file2:
                    scaler = pickle.load(file2)
                with open("C:/Users/Raghavendra Kumar JR/Dump/ohe1.pkl", "rb") as file3:
                    ohe = pickle.load(file3)
                with open("C:/Users/Raghavendra Kumar JR/Dump/ohe2.pkl", "rb") as file4:
                    ohe2 = pickle.load(file4)
                with open("C:/Users/Raghavendra Kumar JR/Dump/oe.pkl", "rb") as file5:
                    oe = pickle.load(file5)
                data = np.array([[np.log(float(quantity_tons)),float(customer),float(country),float(application),np.log(float(thickness)),float(width),status,item_type,int(prod_ref)]])
                data_s = scaler.transform(data[:,:6])
                data_ohe1 = ohe.transform(data[:,[7]])
                # data_ohe2 = ohe2.transform(data[:,[8]])
                data_oe = oe.transform(data[:,[6]])
                data = np.concatenate((data_s,data_ohe1,data_oe), axis=1)
                pred = model.predict(data)
                st.write("Predicted Selling Price is:",np.exp(pred[0]))               

            else:
                st.error("Please enter all values")
with tab2:
    col3,col4 = st.columns(2)
    with col3:
        quantity_tons2 = st.text_input("Quantity tons (Min:0, Max:1000000000):")
        customer2 = st.text_input("Customer (Min:12458, Max:30408185):")
        country2 = st.selectbox("Country:",(0,28,25,30,32,38,78,27,77,113,79,26,39,40,84,80,107,89))
        application2 = st.selectbox("Application:",(0,10,41,28,59,15,4,38,56,42,26,27,19,20,66,29,22,40,25,67,79,3,99,2,5,39,69,70,65,58,68))
        thickness2 = st.text_input("Thickness: (Min:0.18 & Max:400):")
        width2 = st.text_input("Width: (Min:1, Max:2990):")
    with col4:
        selling_price = st.text_input("Selling Price: (Min:1, Max:100001015):")
        item_type2 = st.selectbox("Item type:",("W","WI","S","Others","PL","IPL","SLAWR"))
        prod_ref2 = st.selectbox("Product Reference:",(1670798778,1668701718,628377,640665,611993,1668701376,164141591,1671863738,1332077137,640405,1693867550,1665572374,1282007633,1668701698,628117,1690738206,628112,640400,1671876026,164336407,164337175,1668701725,1665572032,611728,1721130331,1693867563,611733,1690738219,1722207579,929423819,1665584320,1665584662,1665584642))
        predict_status = st.button("Predict Status")
        if predict_status:
            if quantity_tons and customer and thickness and width and selling_price is not None:
                with open("C:/Users/Raghavendra Kumar JR/Dump/model_c.pkl", "rb") as file:
                    model_c = pickle.load(file)
                with open("C:/Users/Raghavendra Kumar JR/Dump/scale_c.pkl", "rb") as file2:
                    scaler_c = pickle.load(file2)
                with open("C:/Users/Raghavendra Kumar JR/Dump/ohe_c.pkl", "rb") as file3:
                    ohe_c = pickle.load(file3)
                data_c = np.array([[np.log(float(quantity_tons)),float(customer),float(country),float(application),np.log(float(thickness)),float(width),np.log(float(selling_price)),item_type,int(prod_ref)]])
                data_c_s = scaler_c.transform(data_c[:,:7])
                data_c_ohe = ohe_c.transform(data_c[:,[7]])
                data_c = np.concatenate((data_c_s,data_c_ohe), axis = 1)
                pred_c = model_c.predict(data_c)
                st.write("Predicted Status is:", pred_c[0])