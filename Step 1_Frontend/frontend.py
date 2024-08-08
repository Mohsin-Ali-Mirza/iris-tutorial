import streamlit as st
import numpy as np
import pickle
import joblib
from joblib import load

def display_iamge(prediction):
    file_path = f"images/{prediction}.jpg"
    st.image(file_path)

def predict_flower(data_inference):

    pickle_file_path = "xgb_model.pkl"
    joblib_file_path = "xgb_model.joblib"

    with open(pickle_file_path,"rb") as f:
        pickle_model = pickle.load(f)
    
    joblib_model = load(joblib_file_path)

    predict_pickle = pickle_model.predict(data_inference)[0]
    predict_joblib = joblib_model.predict(data_inference)[0]



    st.write("The predicition form Pickle:",predict_pickle)
    display_iamge(prediction=predict_pickle)
    st.write("The prediction from JobLib:",predict_joblib)
    display_iamge(prediction=predict_joblib)


def main():
    st.title("Iris Classification")

    sepalLength = st.number_input("Enter Sepal Length in Cm", min_value=0.0,max_value=10.0, value=0.0)
    sepalWidth =  st.number_input("Enter Sepal Width in Cm", min_value=0.0,max_value=10.0, value=0.0)

    petalLength = st.number_input("Enter Petal Length in Cm", min_value=0.0,max_value=10.0, value=0.0)
    petalWidth = st.number_input("Enter Petal Width in Cm", min_value=0.0,max_value=10.0, value=0.0)

    data_inference = np.array([[sepalLength,sepalWidth,petalLength,petalWidth]])

    if(st.button("Predict",type="primary")):
        predict_flower(data_inference=data_inference)
    

if __name__ == "__main__":
    main()