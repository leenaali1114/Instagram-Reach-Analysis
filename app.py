import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import streamlit as st


df = pd.read_csv("Instagram.csv")
df = df.dropna()

X = df[["Likes", "Saves","Comments","Shares","Profile Visits","Follows"]]


y = df["Impressions"]

clf = LogisticRegression() 
clf.fit(X, y)

joblib.dump(clf, "clf.pkl")

st.header("Instagram Reach Prediction App")

like = st.number_input("Enter the number of Likes")
save = st.number_input("Enter the number of Saves")
comm = st.number_input("Enter the number of Comments")
shar = st.number_input("Enter the number of Shares")
prof = st.number_input("Enter the number of Profile Visits")
foll = st.number_input("Enter the number of Followers gained")

if st.button("Submit"):
    clf = joblib.load("clf.pkl")
    X = pd.DataFrame([[like, save, comm, shar, prof, foll]], 
                     columns = ["Likes", "Saves", "Comments", "Shares", "Profile Visits", "Follows"])
                     
    prediction = clf.predict(X)[0]
                     
    st.text(f"The predicted Impression is {prediction}")