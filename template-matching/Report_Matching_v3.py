import os
import ast
import openai
import pandas as pd
import streamlit as st
from openai import AzureOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = "6ad08c8e58dd4de9985b86b1209d9cc2"
os.environ["OPENAI_API_BASE"] = "https://azureopenaitext.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-09-15-preview"
openai.api_type = "azure"
openai.azure_endpoint = "https://azureopenaitext.openai.azure.com/"
openai.api_version = "2023-09-15-preview"
openai.api_key = "6ad08c8e58dd4de9985b86b1209d9cc2"

def create_list_of_strings_from_a_DF(input_df):
    combined_strings = []

    for index, row in input_df.iterrows():
        combined_string = '-'.join(row.astype(str))  # Combine values in the row with '-'
        combined_strings.append(combined_string)

    return combined_string

def create_list_of_strings_of_a_column(df_column):
    combined_strings = [f"{index}: {value}" for index, value in enumerate(df_column)]#, start=1)]
    #combined_strings = df_column.tolist()
    return combined_strings

def result_df(result):
    data_dict = {'Index': [], 'Score': []}  # Initialize an empty dictionary to store data

    # Assuming 'results' contains the list of tuples you've provided

    for item in result:
        # Extract index number and score
        index = int(item[0].page_content.split(':')[0])
        score = item[1]
        
        # Append index number and score to the dictionary
        data_dict['Index'].append(index)
        data_dict['Score'].append(score)
        
    scores_df = pd.DataFrame(data_dict)
    scores_df = scores_df.sort_values('Index')
    return scores_df


def openai_response(query,table):
    response = client.chat.completions.create(
    model="gpt-35-turbo", # model = "deployment_name".
    messages=[
        {"role": "system", "content": f"""Given a DataFrame: {table} and a query: {query}, 
        Return the top 5 row numbers in {table} that best match {query}.
        The output should be a Python list containing only the matching row numbers.
        The row numbers should be aligned with the {table} index.
        Do not include any explanation or text in the output other than a python list of matching rows.
        """},
        ],
    temperature=0.3
    )
    
    return response.choices[0].message.content

def openai_response_2(query,table):
    response = client.chat.completions.create(
    model="gpt-35-turbo", # model = "deployment_name".
    messages=[{"role": "system", "content": f"""Given a DataFrame: {table} and a query: {query}, 
        Return the top 3 row numbers in {table} that best match {query}.
        The output should be a Python list containing only the matching row numbers.
        If there is no match with any row of {table} with {query}, return empty list.
        Do not include any explanation or text in the output other than a python list of matching rows.
        """},
        
        ],
    temperature=0.3
    )
    
    return response.choices[0].message.content

st.set_page_config(layout="wide")
st.title("BI Report Template Matching")
#left_column, right_column = st.columns(2)
query = st.text_input('Enter a Query')
submit_button = st.button('Submit')
#left_column, right_column = st.columns([3,1])#

if submit_button:

    combined_df = pd.read_csv("Merged_MasterData_4Rep.csv")
    links_df = pd.read_csv("Merged_MasterData_Links_4Rep.csv")
    tables_df = pd.read_csv("Merged_Tables_Schema_4Rep.csv")


    embeddings = AzureOpenAIEmbeddings()
    embeddings_input = create_list_of_strings_of_a_column(combined_df['Description'])
    document_search = FAISS.from_texts(embeddings_input,embeddings)

    df_list = []
    new_df = combined_df.copy()#[['Report_Name', 'Page_Name', 'Title']].copy()
    result = document_search.similarity_search_with_score(query,len(embeddings_input))
    new_df['Score'] = result_df(result)['Score'].values
    df_list.append(new_df)
    context_df = df_list[0].nsmallest(10,'Score')

    client = AzureOpenAI(
        api_key = openai.api_key,
        api_version = openai.api_version,
        azure_endpoint = openai.azure_endpoint
    )
    if context_df["Score"].iloc[0] <= 0.45:
        response_rows = openai_response(query, context_df.drop('Score',axis='columns'))
        try:
            result_list = ast.literal_eval(response_rows)
            matching_rows = context_df.loc[result_list]#.reset_index(drop=True)
            if len(matching_rows)>0:
                matching_rows_with_links = pd.merge(matching_rows, links_df[['Report_Name', 'Page_Name', 'Visual_Title', 'Links']], 
                        on=['Report_Name', 'Page_Name', 'Visual_Title'], how='left')
                matching_rows_with_links.drop(["Visual_Type", "Columns", "Description"], axis = 1, inplace = True)
                st.write("")
                st.write("The matching reports are as follows")
                matching_rows_with_links["Score"] = 1 - matching_rows_with_links["Score"]
                #left_area = left_column.write(matching_rows_with_links.drop("Links", axis = 1))
                st.write(matching_rows_with_links)
            else:
                st.write("No matching reports, tables and columns found")
            #right_area = right_column.write(matching_rows_with_links["Links"].iloc[0])
        except:
            st.write("No matching reports, tables and columns found")
    else:
        response_rows_2 = openai_response_2(query, tables_df)
        try:
            result_list_2 = ast.literal_eval(response_rows_2)
            matching_rows = tables_df.loc[result_list_2]#.reset_index(drop=True)
            if len(matching_rows)>0:
                st.write("No Exact matching report found and below are the few suggestion tables")
                st.write(matching_rows.reset_index(drop = True))
            else:
                st.write("No matching reports, tables and columns found")
        except:
            st.write("No matching reports, tables and columns found")
