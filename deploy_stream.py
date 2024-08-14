import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
import streamlit as st


st.title("Prédiction sur les comptes bancaires")
st.subheader('Application réalisée par Lahad')
st.markdown("cette application utilise le model machine learning pour prédire quelles personnes sont les plus susceptibles d’avoir ou d’utiliser un compte bancaire.")


df = pd.read_csv('Financial_inclusion_dataset.csv')
if st.button('Voir notre DataFrame'):
    st.write(df)
    
if st.button('Voir les informations générale de notre DataFrame'):
    st.write(df.info())

if st.button('Voir la statistique de notre DataFrame'):
    st.write(df.describes())

var_num = df.select_dtypes(include=[np.number]).columns

var_cat = df.select_dtypes(include=[object]).columns
var_cat = var_cat.drop('bank_account')

df['bank_account'].replace(['Yes','No'], [1,0], inplace=True)

df[var_num] = df[var_num].fillna(df[var_num].mean())
df[var_cat] = SimpleImputer(strategy='most_frequent').fit_transform(df[var_cat])

if st.button('Vérifier les valeurs manquantes'):
    st.write(df.isnull().sum().sum())

# Suppression des valeurs manquantes
Q1 = df[var_num].quantile(0.25)
Q3 = df[var_num].quantile(0.75)
IQR = Q3 - Q1

born_inf = Q1 - 1.5 * IQR
born_sup = Q3 + 1.5 * IQR

df = df[~((df[var_num] < born_inf) | (df[var_num] > born_sup)).any(axis=1)]

st.subheader('visualisation des valeurs aberrantes aprés sppression')
figure = plt.figure(figsize=(10,7))
sns.boxplot(data=df)
plt.xticks(rotation=90)
st.pyplot(figure)

for col in var_cat:
    df[col] = LabelEncoder().fit_transform(df[col])
    
x= df.drop('bank_account',axis=1)
y= df['bank_account']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Normalisation
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# entraînement de notre model
model_rf = RandomForestClassifier()
model_rf.fit(x_train,y_train)

# La précision 
score_rf = model_rf.score(x_test,y_test)

st.markdown('La précision est:')
st.write(score_rf)


#Définition d'une fonction
def caracteristique(country, year, uniqueid, location_type, cellphone_access, 
                    household_size, age_of_respondent, gender_of_respondent,
                    relationship_with_head, marital_status, education_level, job_type):
    
    
    data = np.array([country, year, uniqueid, location_type, cellphone_access, 
                    household_size, age_of_respondent, gender_of_respondent,
                    relationship_with_head, marital_status, education_level, job_type])
    pred = model_rf.predict(data.reshape(1,-1))
    return pred

st.subheader('Veillez saisir une valeur pour chaque varible')
# Saisissez une valeur pour chaque caracteristique de l'appartement
education_level = st.number_input(label='education_level: ',min_value=0)
job_type = st.number_input(label='job_type: ',min_value=0)
country = st.number_input(label='country: ',min_value=0)
year = st.number_input(label='year: ',min_value=0)
uniqueid = st.number_input(label='uniqueidlocation_type: ',min_value=0)
location_type = st.number_input(label='location_type: ',min_value=0)
cellphone_access= st.number_input(label='cellphone_access : ',min_value=0)
household_size = st.number_input(label='household_size: ',min_value=0)
age_of_respondent = st.number_input(label='age_of_respondent: ',min_value=0)
gender_of_respondent = st.number_input(label='gender_of_respondent: ',min_value=0)
relationship_with_head = st.number_input(label='relationship_with_head: ',min_value=0)
marital_status = st.number_input(label='marital_status: ',min_value=0)


# Création du button 'Predict' qui retourne les prédiction du model
if st.button('Predict'):
    prediction = caracteristique(country, year, uniqueid,location_type,cellphone_access,
                                 household_size, age_of_respondent, gender_of_respondent,
                                 relationship_with_head, marital_status, education_level,
                                 job_type)
    
    resultat=prediction[0]
    st.write(resultat)
    
    if resultat == 1:
        st.success(" Cette personne utilise un compte bancaire")
        st.balloons()
    else: 
        st.warning("Cette personne n'utilise pas de compte bancaire")
    
    
