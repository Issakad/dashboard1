import streamlit as st
from PIL import Image
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt


# Logo "Prêt à dépenser"
    
image = Image.open('img/logo.PNG')
st.sidebar.image(image, width=150)
# Glossaire des features utilisées:




# Définir le glossaire
glossary = {
    "CREDIT_INCOME_PERCENT": "Pourcentage du montant du crédit par rapport au revenu d'un client",
    "ANNUITY_INCOME_PERCENT": "Pourcentage de la rente de prêt par rapport au revenu d'un client",
    "CREDIT_TERM": "Durée du paiement en mois",
    "DAYS_EMPLOYED_PERCENT": "Pourcentage des jours employés par rapport à l'âge du client",
    "NAME_INCOME_TYPE_Businessman": "Revenu de provenant des activités d'affaires",
    "NAME_EDUCATION_TYPE_Higher education":"Niveau d'éducation supérieur",
    "NAME_EDUCATION_TYPE_Incomplete higher": "Education supérieure non achevée",
    "NAME_EDUCATION_TYPE_Lower secondary": "Niveau d'éducation collège",
    "ANNUITY_INCOME_PERCENT": "Montant du remboursement du prêt (annuitées) en % du revenu du client",
    "EXT_SOURCE_1": "Note du client par rapport à son historique de prêts auprès d'autres banques (1)", 
    "EXT_SOURCE_2":"Note du client par rapport à son historique de prêts auprès d'autres banques (2)",
    "EXT_SOURCE_3": "Note du client par rapport à son historique de prêts auprès d'autres banques (3)",
    "DAYS_EMPLOYED_ANOM":"Nombre  jours entre le début du contrat d'emploi du client et la date de sa demande de crédit"
}

# Ajouter une option pour afficher/cacher le glossaire
glossary_visibility = st.sidebar.checkbox("Afficher/Cacher le glossaire")

# Si l'option "Afficher/Cacher" est cochée, afficher le glossaire
if glossary_visibility:
    # Afficher le glossaire en liste déroulante
    selected_term = st.sidebar.selectbox("Sélectionnez un terme :", list(glossary.keys()))

    # Afficher la définition du terme sélectionné
    if selected_term in glossary:
        st.sidebar.write(f"## {selected_term}")
        st.sidebar.write(glossary[selected_term])
    else:
        st.sidebar.write("Sélectionnez un terme dans la liste déroulante.")
else:
    st.sidebar.write("Le glossaire est caché. Cochez la case 'Afficher/Cacher le glossaire' pour le voir.")

def main():

    #API_URL = "http://127.0.0.1:5000/api/"
    API_URL = "https://issakaapi.herokuapp.com/api/"

    

    st.title("Dashboard pour l'octroi de crédits")
    st.subheader("Auteur: Issaka Dialga")


    #################################################
    # LIST OF SK_ID_CURR
    # Get list of SK_IDS (cached)
    @st.cache_data
    def get_sk_id_list():

        # URL of the sk_id API
        SK_IDS_API_URL = API_URL + "sk_ids/"

        # Requesting the API and saving the response
        response = requests.get(SK_IDS_API_URL)
        #print(response)

        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        #print(content)

        # Getting the values of SK_IDS from the content
        SK_IDS = content['data']

        return SK_IDS
    SK_IDS = get_sk_id_list()

     ##################################################
    # Selecting applicant ID
    select_sk_id = st.sidebar.selectbox('Sélectionné un client:', SK_IDS, key=1)
    st.write('Vous avé sélectionné le client: ', select_sk_id)
    
    
    ##################################################
    # SCORING
    ##################################################
    st.header('Score du client sélectionné et Décision de la banque')

    # Get scoring (cached)
    @st.cache_data
    def personal_scoring(select_sk_id):
        # URL of the scoring API
        SCORING_API_URL = API_URL + "scoring/?SK_ID_CURR=" + str(select_sk_id)

        # Requesting the API and save the response
        response = requests.get(SCORING_API_URL)

        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))

        # getting the values from the content
        
        
        
        score = content['score']

        return score
        
    

     # Get local interpretation of the score (surrogate model, cached)
    @st.cache_data
    def score_explanation(select_sk_id):
        # URL of the scoring API
        SCORING_EXP_API_URL = API_URL + "local_interpretation?SK_ID_CURR=" + str(select_sk_id)

        # Requesting the API and save the response
        response = requests.get(SCORING_EXP_API_URL)

        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))

        # getting the values from the content
        prediction = content['prediction']
        
        contribs =  pd.Series(content['contribs']).rename("Feature contributions")

        return (prediction, contribs)



    if st.sidebar.checkbox('Afficher le score du client sélectionné'):
        # Get score
        score = personal_scoring(select_sk_id)
        # Display score (default probability)
        st.write('Le score du client sélectionné est:', score, '%')
        if score < 10:
            st.write("Le client a de bonnes chances de rembourser son prêt.")
        elif score < 20:
            st.write("Le client a des chances moyennes de rembourser son prêt.")
        else:
            st.write("Le client a très peu de chances de rembourser son prêt." )
        st.write("Décision de la banque :")
        if score< 10:
            st.write("Prêt accordé !👋✍️🤝") 
        else:
            st.write("Crédit refusé.😕😕😕")

        if st.checkbox('Afficher les informations expliquant le score du client'):
            # Get prediction, bias and features contribs from surrogate model
            (_, contribs) = score_explanation(select_sk_id)
            # Display the bias of the surrogate model
            #st.write("Population mean (bias):", bias*100, "%")
            # Remove the features with no contribution
            contribs = contribs[contribs!=0]
            # Sorting by descending absolute values
            contribs = contribs.reindex(contribs.abs().sort_values(ascending=False).index)

            st.dataframe(contribs)
            st.bar_chart(contribs)
            


    ##################################################
    # FEATURES' IMPORTANCE
    ##################################################
    st.header('Interprétation globale')

    # Get features importance (surrogate model, cached)
    @st.cache_data
    def get_features_importance():
        # URL of the features' importance API
        FEATURES_IMP_API_URL = API_URL + "features_imp"
    
        # save the response to API request
        response = requests.get(FEATURES_IMP_API_URL)
        
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))

        # convert data to pd.Series
        features_imp = pd.Series(content['data']).rename("Features importance").sort_values(ascending=False)
        features_imp_df = pd.DataFrame(features_imp)

        return features_imp
    
    if st.sidebar.checkbox("Affichez le graphique de l'interprétation globale"):
        # Get the features' importance
        features_imp = get_features_importance()

        # Initialization
        sum_fi = 0
        labels = []
        frequencies = []

        # Get the labels and frequencies of features
        for feat_name, feat_imp in features_imp.items():
            labels.append(feat_name)
            frequencies.append(feat_imp)
            sum_fi += feat_imp

    # Complete the FI of other features
    #labels.append("OTHER FEATURES...")
    #frequencies = []
    #sum_fi = 0
    #labels = []

        frequencies.append(max(0, 1 - sum_fi))  # Ensure non-negative value

        # Set up the figure and axes
        fig, ax = plt.subplots()
        ax.axis("equal")
        ax.pie(frequencies, autopct='%1.1f%%')  # Display frequencies as percentages
        ax.set_title("Features importance")
        ax.legend(
        labels,
        loc='center left',
        bbox_to_anchor=(0.9, 0.5),
    )

        # Plot the pie chart using Streamlit
        st.pyplot(fig)
        # Display the bar chart of features_imp
        st.bar_chart(features_imp)
        


    #if st.checkbox('Montrez les  details des features importances'):
       # st.dataframe(features_imp)


        ##################################################
    # PERSONAL DATA
    ##################################################
    st.header('Interprétation locale')

    # Personal data (cached)
    @st.cache_data
    def get_local_interpretation():
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        LOCAL_DATA_API_URL = API_URL + "local_interpretation/?SK_ID_CURR=" + str(select_sk_id)

        # save the response to API request
        response = requests.get(LOCAL_DATA_API_URL)

        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))

        # convert data to pd.Series
        #personal_data = pd.Series(content['data']).rename("SK_ID {}".format(select_sk_id))
        local_data= pd.Series(content['contribs']).rename("Local_contribution").sort_values(ascending=False)

        return local_data
    if st.sidebar.checkbox('Affichez les features importances pour le client sélectionné'):
        # Get personal data
        local_data= get_local_interpretation()
        st.dataframe(local_data)
        st.bar_chart(local_data)
        
        

    
    
    
     # Get data from 20 nearest neighbors in train set (cached)
    @st.cache_data
    def get_data_neigh():
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        NEIGH_DATA_API_URL = API_URL + "neigh_cust/?SK_ID_CURR=" + str(select_sk_id)
        # save the response of API request
        response = requests.get(NEIGH_DATA_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame and pd.Series
        X_neigh = pd.DataFrame(content['X_neigh'])
        y_neigh = pd.Series(content['y_neigh']['TARGET']).rename('TARGET')
        return X_neigh, y_neigh
    


    #if st.sidebar.checkbox('Afficher les features importances des clients similaires'):

        # Get personal data
        #personal_data = get_data_neigh(select_sk_id)

        
            # Get 20 neighbors personal data (preprocessed)
        X_neigh, y_neigh = get_data_neigh()
        y_neigh = y_neigh.replace({0: 'client solvable (neighbors)',
                               1: 'client non solvable (neighbors)'})
            # Concatenation of the information to display
        df_display = pd.concat([X_neigh, y_neigh], axis=1)

        
        
        st.dataframe(df_display)


       
    

    ##################################################
    # FEATURES DESCRIPTIONS
    ##################################################
    st.header("Comapraison des informations du client sélectionné et des clients similaires") 
    # Get the list of features
    @st.cache_data
         

    def get_features_descriptions(select_sk_id):
        
        SK_IDS = get_sk_id_list() # Replace with your list of SK_ID_CURR values
        
        FEAT_DESC_API_URL = API_URL + "features_desc/?SK_ID_CURR=" + str(select_sk_id)
        response = requests.get(FEAT_DESC_API_URL)
        content = json.loads(response.content.decode('utf-8'))
        data = content['data']
        features_desc = pd.DataFrame(data)
        features_desc_transposed = features_desc.transpose()
        
        return features_desc_transposed

    # Select the SK_ID_CURR from the list
    select_sk_id = st.sidebar.selectbox('Sélectionné un client:', SK_IDS, key=2)


    if st.sidebar.checkbox('Affichez les statistiques descriptives du client sélectionné et des clients similaires'):
        # Display features' descriptions
        features_desc = get_features_descriptions(select_sk_id)
        st.table(features_desc)
        
        # Extract the relevant columns from the features_desc dataframe
        statistique = features_desc.columns[2:]
        client_mean = pd.to_numeric(features_desc.iloc[0, 2:].values, errors='coerce')
        similar_clients_mean = pd.to_numeric(features_desc.iloc[1, 2:].values, errors='coerce')
        
        # Create a DataFrame with the mean values
        data = {'Features': statistique,
                'Client sélectionné': client_mean,
                'Clients similaires': similar_clients_mean}
        df = pd.DataFrame(data)
        
        # Set the column 'Statistique' as the index
        df.set_index('Features', inplace=True)
        
        # Create the bar chart
        fig, ax = plt.subplots()
        df.plot(kind='bar', ax=ax)
        ax.set_ylabel('Values')
        ax.set_title('Comparaison des informations du client sélectionné et des clients similaires')
        
        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=45)
        
        # Display the plot
        st.pyplot(fig)



    



    ################################################


if __name__== '__main__':
    main()
