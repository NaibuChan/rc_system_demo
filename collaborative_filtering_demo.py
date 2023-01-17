import streamlit as st
import pandas as pd 
import pickle 

# Chia layout
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
# Tách làm đôi 5 5
col1, col2 = st.columns((5,5))

# Load data và model
final_model = pickle.load(open('E:\\h-and-m-personalized-fashion-recommendations\\collaborative_model.sav', 'rb'))
meta_articles = pd.read_csv('E:\\h-and-m-personalized-fashion-recommendations\\articles.csv', index_col='article_id')
transactions = pd.read_csv('E:\\h-and-m-personalized-fashion-recommendations\\cf_dataset.zip')
df_customer = transactions.set_index('customer_id')
df_customer.drop(columns=['InvoiceDate', 'price', 'sales_channel_id', 't_dat', 'date', 'bought'], inplace=True)  # Bỏ vì không cần thiết nữa

def customer_article_recommend(user, n_recs):
    have_bought = list(df_customer.loc[user]['article_id'])

    not_bought = meta_articles.copy()
    not_bought.drop(have_bought, inplace=True)
    not_bought.reset_index(inplace=True)
    not_bought['est_purchase'] = not_bought['article_id'].apply(lambda x: final_model.predict(user, x).est)
    not_bought.sort_values(by='est_purchase', ascending=False, inplace=True)
    not_bought.rename(columns={'prod_name':'Product Name','product_type_name':'Product Type Name', 'product_group_name':'Product Group Name',
                               'index_group_name':'Index Group Name', 'garment_group_name':'Garment Group Name'}, inplace=True)
    not_bought = not_bought[['article_id','Product Name', 'Product Type Name', 'Product Group Name', 'Index Group Name', 'Garment Group Name']]
    not_bought = not_bought.iloc[:100, :]
    not_bought = not_bought.sample(frac=1).reset_index(drop=True)
    return not_bought.head(n_recs)

# Bên phải
with col1:
    st.title("Collaborative Filtering Recommender System")
    #Tạo form điền input
    with st.form('Form'):
        n_recs = st.number_input("Number of recommendation", value = 10)
        user = st.text_input('Customer\'s ID')
        submitted = st.form_submit_button('Submit')
        
    # Nếu bấm vào nút Submit
    if(submitted):
        st.text('Top {} recommendations for customer: {}'.format(n_recs ,user))
        st.dataframe(customer_article_recommend(user, n_recs))

# Bên trái
with col2:
    st.dataframe(df_customer.index.unique(),)

