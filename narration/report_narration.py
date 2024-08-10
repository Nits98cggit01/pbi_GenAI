import streamlit as st
import pandas as pd
import os
import re
import urllib.parse
import math
import shutil
import openai
import base64
import glob
import time

from pathlib import Path
from pdf2image import convert_from_path
from PyPDF2 import PdfReader,PdfWriter
from PIL import Image  # Import Image from PIL
from fpdf import FPDF
from PyPDF2 import PdfMerger
from urllib.parse import urlparse
from config import *

from langchain.agents.agent_types import AgentType
# from langchain_experimental.agents.agent_toolkits import create_csv_agent
# from langchain_openai import ChatOpenAI, OpenAI, AzureOpenAI
from langchain.llms import OpenAI,AzureOpenAI
from langchain_community.llms import OpenAI, AzureOpenAI

# Helper functions

def store_report_details(url,name):
    with open(os.path.join(ROOT_PATH,"details.txt"), 'w') as details_txt:
        details_txt.write(f'{url} \n{name}')

def extract_pages():
    csv_files = [f for f in os.listdir(read_folder) if f.endswith('.csv')]
    sheet_names = set()
    for file in csv_files:
        sheet_name = file.split('_')[-2]  
        print(f'Page name : {sheet_name}')
        sheet_names.add(sheet_name)
        
    return sorted(list(sheet_names))  

def extract_visuals(): 
    csv_files = [f for f in os.listdir(visual_data_folder) if f.endswith('.csv')]
    metrics_names = set()
    for file in csv_files:
        metrics_name = file.split('_')[-1]
        metrics_name = metrics_name.split('.')[0]
        metrics_names.add(metrics_name)
        print(f'metrics_names : {metrics_names}') 
 

    return sorted(list(metrics_names))

def get_details(selected_page):
    csv_folder_path = read_folder 
    output_folder = visual_data_folder    
    #src_path = os.path.join(analysis_folder,f'{report_name}_MasterData.csv')
    src_path = os.path.join(folder_path,f'{report_name}_MasterData.csv')
    
    print(f'Selected page is : {selected_page}')
    src_df = pd.read_csv(src_path)
    if "Unnamed: 0" in src_df:
                src_df.drop(columns = "Unnamed: 0", axis = 1, inplace = True)
        #src_df.to_csv(os.path.join(analysis_folder,f"{report_name}_MasterData.csv"))
    src_df.to_csv(os.path.join(folder_path,f"{report_name}_MasterData.csv"))
    
    if selected_page == 'All Pages':    
        csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
        for file in csv_files:
            print(f'Check : {file}')
            shutil.copy(os.path.join(csv_folder_path, file), output_folder)

    else:
        csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]        
        for file in csv_files:
            print(f'The files present : {file}')
            # sheet_name = file.split('_')[-2]
            # consolidated_sheetname = sheet_name.split(' ')[0]
            consolidated_sheetname = file.split('_')[-2]
            print(f'Consolidated_sheet : {consolidated_sheetname}')
            if selected_page == consolidated_sheetname:
                shutil.copy(os.path.join(csv_folder_path, file), output_folder)

def call_openai(prompt):
    openai.api_type = API_TYPE
    openai.api_base = API_BASE
    openai.api_version = API_VERSION
    openai.api_key = API_KEY

    message_text = [{"role":"system","content":f"{prompt}"}]

    response = openai.ChatCompletion.create(
          engine="gpt-35-turbo",
          messages = message_text,
          temperature=0.2,
          max_tokens=500,
          top_p=0.95,
          frequency_penalty=0,
          presence_penalty=0,
          stop=None
        )

    res_output = response['choices'][0]['message']['content'] 
    return res_output

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def filter_insights():
    if selected_metrics == 'All Metrics':
        visual_files = os.listdir(visual_data_folder) 
        visual_base_names = {os.path.splitext(file)[0] for file in visual_files}
        # Get the list of files in the insights folder and extract base names
        insights_files = os.listdir(insights_folder)
        insights_base_names = {os.path.splitext(file)[0] for file in insights_files}
        # Find the common base names
        common_base_names = visual_base_names & insights_base_names

        # Copy the matching files from insights to analysis folder
        for file in insights_files:
            base_name = os.path.splitext(file)[0]
            print(f'The base name : {base_name}')
            if base_name in common_base_names:
                src_path = os.path.join(insights_folder, file)
                dest_path = os.path.join(analysis_folder, file)
                shutil.copyfile(src_path, dest_path)
                print(f"Copied {file} to {analysis_folder}")
    
    else:
        insights_files = os.listdir(insights_folder)
        insights_base_names = {os.path.splitext(file)[0] for file in insights_files}
        required_file = f'{report_name}_{selected_page}_{selected_metrics}'
        for file in insights_files:
            base_name = os.path.splitext(file)[0]
            print(f'The base name : {base_name}')
            if base_name in required_file:
                src_path = os.path.join(insights_folder, file)
                dest_path = os.path.join(analysis_folder, file)
                shutil.copyfile(src_path, dest_path)
                print(f"Copied {file} to {analysis_folder}")

def report_generation(analysis_folder,section_prompt):
    filter_insights()
    print(f'Folder path : {analysis_folder}')    
    all_files = os.listdir(analysis_folder)
    txt_files = [f for f in all_files if f.endswith('.txt')]
    print(f"Insight file list : {txt_files}")
    #if not os.path.exists(response_folder):
    #    os.makedirs(response_folder)
        
    for index,file_name in enumerate(txt_files):
        file_path = os.path.join(insights_folder, file_name)
        print(file_name)
        file_parts = os.path.splitext(file_name)[0].split('_')

        sheet_name = file_parts[-2]
        viz_name = file_parts[-1]
        ip_content = read_txt_file(file_path)

        gpt_prompt = f"""
        Act as a Data analyst. Your task is to generate report using the following information. 
        The input content is :
        {ip_content}       
        {section_prompt}
        """
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print(f'The prompt given for LLM is : {gpt_prompt}')
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        
        res_output = call_openai(gpt_prompt)
        print(res_output)
        output_file_path = os.path.join(llm_response_folder,f'{sheet_name}_{viz_name}.txt')
        with open(output_file_path,'w') as file:
            file.write(f"""
Report name : {report_name}\n
Sheet name : {sheet_name}\n
Title : {viz_name}\n

{res_output}
            """)
        
        print("##########OUTPUT GENERATED#########")
        
    return res_output

def ask_bot():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    # Try
     
    #####
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        input_content = read_txt_file(os.path.join(insights_folder,f'{report_name}_{selected_page}_{selected_metrics}.txt'))
        sample_prompt = f'''Act as a Data analyst, you have a table given in a txt format, the content is \n
        {input_content}    
    {prompt}
    Provide only the answer and not any other supporting statements. Strictly do not answer for any generic questions which is irrelavant to the input above!!!
    STRICTLY DO NOT ANSWER TO ANY QUESTION WHICH IS IRRELAVENT TO THE ABOVE CONTENT!!!
        '''
        
        get_bot_response = call_openai(sample_prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(get_bot_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": get_bot_response})

def call_csv_agent():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    # Try
     
    #####
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # get_agent_response = trigger_csv_agent(prompt)
        # Display assistant response in chat message container
        # with st.chat_message("assistant"):
        #     st.markdown(get_agent_response)
        # Add assistant response to chat history
        display_agent_answer = langchain_csv_agent(prompt)
        print(f'The final answer is : {display_agent_answer}')
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(display_agent_answer)
        st.session_state.messages.append({"role": "assistant", "content": display_agent_answer})

def langchain_csv_agent(question):
    qa_result_file = os.path.join(ROOT_PATH, "csv_agent_response.txt")
    with open(os.path.join(ROOT_PATH,"csv_agent_question.txt"),'w') as csv_agent_file:
        csv_agent_file.write(question)

    relative_notebook_path = f'data-extraction\\ask_csv_agent.ipynb'
    if os.path.isfile(qa_result_file):
        os.remove(qa_result_file)
    csv_agent_url = create_jupyter_link(ROOT_PATH,relative_notebook_path)
    st.markdown(f"[Click this link to trigger Q&A]({csv_agent_url})")
    
    time.sleep(15)
    print(f'Executing after 5 seconds')
    if os.path.isfile(qa_result_file):
        print(f'The file exists')
        # agent_answer = extract_final_answer(qa_result_file)
        try:
            with open(qa_result_file, 'r') as file:
                content = file.read()

            print(f'The opened file is : {qa_result_file}')
            # Find the start position of 'Final Answer:'
            final_answer_start = content.find('Final Answer:')
            
            if final_answer_start == -1:
                print("Final Answer not found.")
                return

            # Isolate the part of the content starting from 'Final Answer:'
            final_answer_content = content[final_answer_start + len('Final Answer:'):]

            # Find the next occurrences of 'Question:', 'Thought:', 'Action:', 'Action Input:'
            end_markers = ['Question:', 'Thought:', 'Action:', 'Action Input:']
            end_positions = [final_answer_content.find(marker) for marker in end_markers]
            end_positions = [pos for pos in end_positions if pos != -1]

            # Determine the position of the first end marker
            if end_positions:
                end_position = min(end_positions)
                final_answer = final_answer_content[:end_position].strip()
            else:
                final_answer = final_answer_content.strip()

            print(f'This is the final_answer after extraction : {final_answer}')

        except Exception as e:
            print(f"An error occurred: {e}")

    else:
        print(f'The required file does not exist')
        st.write(f'No response from csv agent')
    
    return final_answer


def create_pdf(file_path, output_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False, margin=15)
    pdf.set_font("Times", size=10)

    with open(file_path, 'r', encoding='utf-8') as txt_file:
        content = txt_file.readlines()

        # Extract necessary information
        report_name = content[1].split(' : ')[1].strip()
        sheet_name = content[3].split(' : ')[1].strip()
        title = content[5].split(' : ')[1].strip()

        pdf.add_page()

        # Adding logos
        pdf.image(cg_logo_path, 10, 10, 30)
        
        # Apply border to the entire page
        pdf.set_line_width(1)
        pdf.rect(5.0, 5.0, pdf.w - 10.0, pdf.h - 10.0)
        
        pdf.ln(5)
        pdf.set_font('Times', 'BU', 12)
        pdf.cell(0, 2, f'{sheet_name}:{title}', ln=True, align='C')
        pdf.ln(5)
        
        # Create a table
        pdf.set_fill_color(200)
        pdf.set_text_color(0)
        pdf.set_draw_color(0)
        pdf.set_line_width(0.1)

        #new_content = txt_file.readlines()[12:] 
        new_content = ''.join(content[7:])  # Combining lines into a single string
        sections = new_content.split("\n\n")
        for section in sections:
            lines = section.split("\n")
            for line in lines:
                if line.startswith("Introduction") or line.startswith("Summary") or line.startswith("Statistical analysis") or line.startswith("Conclusion") or line.startswith("General analysis") or line.startswith("Trend analysis") or line.startswith("Impact of Covid19"):
                    pdf.set_font("Times", 'B', 12)
                    pdf.cell(0, 10, line, ln=True)
                else:
                    pdf.set_font("Times", size=10)
                    pdf.multi_cell(0, 5, line)

        
        # Save the PDF file
        pdf.output(output_path)

def getpdf(llm_response_folder, report_folder):
    for file_name in os.listdir(llm_response_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(llm_response_folder, file_name)
            
            pdf_output_path = os.path.join(report_folder, f'{os.path.splitext(file_name)[0]}.pdf')
            create_pdf(file_path, pdf_output_path)

    print(f"PDFs generated with content from text files.")
  
def consolidated_report(report_folder,final_report_name):       
    merger = PdfMerger()   
    if selected_page == 'All Pages':
        input_file_path = os.path.join(folder_path,f"{report_name}_MasterData.csv")
        input_df = pd.read_csv(input_file_path)
        sorted_list = []

        for i in range(0,len(input_df)):
            sorted_list.append(input_df['Title'][i])

        try:
            for title in sorted_list:
                for file_name in os.listdir(report_folder):
                    if title in file_name and file_name.endswith('.pdf'):
                        pdf_path = os.path.join(report_folder, file_name)
                        merger.append(pdf_path)

        except:
            pass

        merger.write(os.path.join(folder_path,final_report_name))
        merger.close()
        # Try compressing 20231312
        try:
            reader = PdfReader(os.path.join(folder_path,final_report_name))
            writer = PdfWriter()

            for page in range(reader.getNumPages()):
                writer.addPage(reader.getPage(page))  
                writer.compressContentStreams()

            with open(os.path.join(folder_path,final_report_name), "wb") as f:
                writer.write(f)

        except Exception as e:
            print(f"Compression failed. Error: {e}")
            
        # End of pdf compress 20231312
        print(f"Master PDF generated at: {os.path.join(folder_path,final_report_name)}")

    else:
        for file_name in os.listdir(report_folder):
            if file_name.endswith('.pdf'):
                print(file_name)
                pdf_path = os.path.join(report_folder, file_name)
                merger.append(pdf_path)

        # Output the merged PDF to the specified path
        merger.write(os.path.join(folder_path,final_report_name))
        merger.close()
        print(f"Master PDF generated at: {os.path.join(folder_path,final_report_name)}")

def display_pdf(selected_page):
    pdf_path = os.path.join(folder_path, f"{report_name}_{selected_page}_{selected_metrics}_{selected_section}.pdf")  # Assuming the PDFs are named after the selected page
    filename = f"{report_name}_{selected_page}_{selected_metrics}_{selected_section}.pdf"
    # Check if 'All Pages' is selected
    if selected_page == 'All Pages':
        st.write(f"The pdf is generated successfully")
        #st.markdown(f'<a href="file://{pdf_path}" target="_blank">{pdf_path}</a>', unsafe_allow_html=True)
        #st.write(pdf_path)
        
        with open(pdf_path,'rb') as f:
            pdf_content = f.read()
        st.download_button(
            label="Download PDF",
            data=pdf_content,
            mime="application/pdf",
            file_name=filename,
            key="download-pdf-btn"
        )
    else:
        # Display the selected page
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')

            pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf" zoom="100%">'
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.write("PDF not found for this page.")

def preprocess_numerical_column(series):
    # Replace parentheses with negative sign and remove any non-numeric characters
    series = series.replace('[\$,]', '', regex=True).replace('[()]', '', regex=True)
    series = series.str.replace(',', '').astype(float)
    return series

def preprocess_percentage_column(series):
    # Remove '%' sign and convert to float
    series = series.str.replace('%', '').astype(float)
    return series

def try_parse_date(column):
    try:
        return pd.to_datetime(column, format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        return column

def generate_insights(read_folder,insights_folder):
    # Ensure the insights folder exists
    os.makedirs(insights_folder, exist_ok=True)
    
    # Process each CSV file in the read_folder
    for file_name in os.listdir(read_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(read_folder, file_name)
            
            # Step 1: Read the CSV file
            df = pd.read_csv(file_path)
            
            # Remove 'Unnamed' columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # Step 2: Parse the filename
            base_name = os.path.basename(file_path)
            name_parts = base_name.split('_')

            if len(name_parts) < 3:
                print(f"Skipping file {file_name} - filename does not follow the expected pattern")
                continue

            visual_name = name_parts[-1].replace('.csv', '')
            page_name = name_parts[-2]
            report_name = '_'.join(name_parts[:-2])

            insights = []

            if df.shape[1] == 1:
                # There's only one column in the DataFrame after removing 'Unnamed' columns
                col_name = df.columns[0]
                insights.append(f"Single Column: {col_name}\n")
                unique_values = df[col_name].unique()
                
                # Attempt to parse as datetime
                parsed_column = try_parse_date(df[col_name])
                if pd.api.types.is_datetime64_any_dtype(parsed_column):
                    df[col_name] = parsed_column
                    df_sorted = df.sort_values(by=col_name)
                    first_date = df_sorted[col_name].iloc[0]
                    last_date = df_sorted[col_name].iloc[-1]
                    num_entries = df_sorted[col_name].count()
                    insights.append(f"This csv contains date field starting from '{first_date}' to '{last_date}' - {num_entries} entries\n")
                elif df[col_name].astype(str).str.contains(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b').any():
                    # Check if the column contains month names
                    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    df_sorted = df[col_name].apply(lambda x: month_order.index(x[:3])).sort_values()
                    first_month = df[col_name].iloc[df_sorted.index[0]]
                    last_month = df[col_name].iloc[df_sorted.index[-1]]
                    num_entries = df[col_name].count()
                    insights.append(f"This csv contains month field from '{first_month}' to '{last_month}' - {num_entries} entries\n")
                elif len(unique_values) == 1:
                    # Single unique value
                    insights.append(f"This csv contains only 1 entry that is {col_name} and the value is {unique_values[0]}\n")
                else:
                    # Count the number of occurrences of each unique value
                    value_counts = df[col_name].value_counts()
                    for value, count in value_counts.items():
                        insights.append(f"{value} - {count}\n")
            elif df.shape[1] == 2:
                # Exactly two columns scenario
                col1, col2 = df.columns
                insights.append(f"Columns: {col1}, {col2}\n")
                unique_entries_col1 = df[col1].unique()

                if pd.api.types.is_numeric_dtype(df[col2]) or (df[col2].dtype == 'object' and (df[col2].str.contains('\$').any() or df[col2].str.contains('%').any())):
                    # If the second column is numerical (considering $ and % as numeric), sum the values grouped by the first column
                    if df[col2].dtype == 'object' and df[col2].str.contains('\$').any():
                        df[col2] = preprocess_numerical_column(df[col2])
                    if df[col2].dtype == 'object' and df[col2].str.contains('%').any():
                        df[col2] = preprocess_percentage_column(df[col2])
                    grouped_sum = df.groupby(col1)[col2].sum().reset_index()
                    insights.append(f"Sum of {col2} grouped by {col1}:\n")
                    insights.append(grouped_sum.to_string(index=False) + "\n")
                else:
                    # If the second column is non-numerical, count occurrences of each unique combination
                    df_counts = df.groupby([col1, col2]).size().reset_index(name='Count')
                    insights.append(f"Count of unique {col1} and {col2} combinations:\n")
                    insights.append(df_counts.to_string(index=False) + "\n")
            else:
                # Multiple columns scenario
                # Identify categorical and numerical fields
                categorical_fields = [col for col in df.select_dtypes(include=['object']).columns]
                numerical_fields = [col for col in df.select_dtypes(include=['number']).columns]
                
                # Check for columns with '$' or '%' sign and preprocess them
                for col in df.columns:
                    if df[col].dtype == 'object' and df[col].str.contains('\$').any():
                        df[col] = preprocess_numerical_column(df[col])
                        numerical_fields.append(col)
                        if col in categorical_fields:
                            categorical_fields.remove(col)
                    if df[col].dtype == 'object' and df[col].str.contains('%').any():
                        df[col] = preprocess_percentage_column(df[col])
                        numerical_fields.append(col)
                        if col in categorical_fields:
                            categorical_fields.remove(col)
                    # Attempt to parse as datetime
                    df[col] = try_parse_date(df[col])
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        if col in categorical_fields:
                            categorical_fields.remove(col)

                unique_entries = {field: df[field].unique() for field in categorical_fields}

                # List of categorical and numerical fields
                insights.append("Categorical columns:\n" + ", ".join(categorical_fields) + "\n")
                insights.append("Numerical columns:\n" + ", ".join(numerical_fields) + "\n")

                # Categorical insights
                for field in categorical_fields:
                    value_counts = df[field].value_counts()
                    insight = f"Columns: {field}\n"
                    for value, count in value_counts.items():
                        insight += f" - {value}: {count}\n"
                    insights.append(insight)

                # Numerical insights grouped by categorical fields
                for cat_field in categorical_fields:
                    for num_field in numerical_fields:
                        grouped_stats = df.groupby(cat_field)[num_field].agg(['count', 'sum', 'mean', 'max', 'min']).reset_index()
                        grouped_stats['sum'] = grouped_stats['sum'].apply(lambda x: f"${x:,.2f}")
                        grouped_stats['mean'] = grouped_stats['mean'].apply(lambda x: f"${x:,.2f}")
                        grouped_stats['max'] = grouped_stats['max'].apply(lambda x: f"${x:,.2f}")
                        grouped_stats['min'] = grouped_stats['min'].apply(lambda x: f"${x:,.2f}")
                        insight = f"Numerical Insights for {num_field} grouped by {cat_field}:\n"
                        insight += grouped_stats.to_string(index=False) + "\n"
                        insights.append(insight)

            # Step 5: Write to a text file
            output_file_path = os.path.join(insights_folder, os.path.splitext(base_name)[0] + '.txt')
            with open(output_file_path, 'w') as f:
                f.write(f"Report Name: {report_name}\n")
                f.write(f"Page Name: {page_name}\n")
                f.write(f"Visual Name: {visual_name}\n\n")
                f.write("Insights:\n")
                for insight in insights:
                    f.write(insight + "\n")

            print(f"Insights written to {output_file_path}")

def remove_unnamed_column_from_folder(visual_data_folder):
    csv_list = []
    try:
        # Iterate through all files in the folder
        for file_name in os.listdir(visual_data_folder):
            if file_name.endswith('.csv'):
                file_path = os.path.join(visual_data_folder, file_name)
                
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Check if 'Unnamed: 0' column exists and drop it
                if 'Unnamed: 0' in df.columns:
                    df.drop(columns=['Unnamed: 0'], inplace=True)
                
                # Save the modified DataFrame back to the CSV file
                df.to_csv(file_path, index=False)
                
                # Add the file name to the list
                csv_list.append(file_path)
        
        print("Successfully processed all CSV files and removed 'Unnamed: 0' columns.")
        print(csv_list)
        return csv_list
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def extract_final_answer(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        # Find the start position of 'Final Answer:'
        final_answer_start = content.find('Final Answer:')
        
        if final_answer_start == -1:
            print("Final Answer not found.")
            return

        # Isolate the part of the content starting from 'Final Answer:'
        final_answer_content = content[final_answer_start + len('Final Answer:'):]

        # Find the next occurrences of 'Question:', 'Thought:', 'Action:', 'Action Input:'
        end_markers = ['Question:', 'Thought:', 'Action:', 'Action Input:']
        end_positions = [final_answer_content.find(marker) for marker in end_markers]
        end_positions = [pos for pos in end_positions if pos != -1]

        # Determine the position of the first end marker
        if end_positions:
            end_position = min(end_positions)
            final_answer = final_answer_content[:end_position].strip()
        else:
            final_answer = final_answer_content.strip()

        print(final_answer)

    except Exception as e:
        print(f"An error occurred: {e}")

# def trigger_csv_agent(question):
#     llm = AzureOpenAI(
#         openai_api_key=API_KEY,
#         deployment_name=OPENAI_DEPLOYMENT_NAME,
#         model_name=MODEL_NAME,
#         azure_endpoint = API_BASE,
#         api_version=API_VERSION,
#         temperature=0
#     )
#     csv_list = remove_unnamed_column_from_folder(visual_data_folder)
#     agent = create_csv_agent(llm, 
#                          csv_list,
#                          verbose=True,
#                         )
#     try:
#         check = agent.invoke(question)
#         with open(os.path.join(folder_path,"csv_agent_response.txt"),'w') as csv_agent_file:
#             csv_agent_file.write(check)
#     except Exception as e:
#         print(e)
#         with open(os.path.join(folder_path,"csv_agent_response.txt"),'w') as csv_agent_file:
#             csv_agent_file.write(str(e))
    
#     agent_response_file = os.path.join(folder_path,"csv_agent_response.txt")
#     agent_response = extract_final_answer(agent_response_file)
#     return agent_response

def isextraction_required(ROOT_PATH,report_name):
    print(f'Root path is : {ROOT_PATH} and report name passed is : {report_name}')
    if report_name != '':
        report_path = os.path.join(ROOT_PATH, report_name)
        extraction_counter = os.path.isdir(report_path)
        print(extraction_counter)
    
    return extraction_counter

def create_jupyter_link(root_path, relative_notebook_path):
    """
    Create a Jupyter Notebook link given the root path and the relative path to the notebook.
    :param root_path: The root directory where the project is located.
    :param relative_notebook_path: The relative path from the root directory to the Jupyter notebook file.
    :return: A string containing the full Jupyter Notebook URL.
    """
    # Combine the root path and the relative path to form the full path to the notebook
    full_notebook_path = os.path.join(root_path, relative_notebook_path)
    full_notebook_path = os.path.normpath(full_notebook_path)
    one_drive_index = full_notebook_path.find('OneDrive - Capgemini')
    
    if one_drive_index == -1:
        raise ValueError("The path does not contain 'OneDrive - Capgemini'")
    
    local_path = full_notebook_path[one_drive_index:]
    url_path = urllib.parse.quote(local_path.replace(os.sep, '/'))
    jupyter_link = f'http://localhost:8888/notebooks/{url_path}'  
    return jupyter_link

def is_valid_powerbi_url(url):
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Check the base domain
    if parsed_url.scheme != 'https' or parsed_url.netloc != 'app.powerbi.com':
        return False
    
    # Split the path and check its components
    path_parts = parsed_url.path.strip('/').split('/')

    if path_parts[0] != 'groups':
        return False
    
    if not path_parts[1]:  # groupid should be non-empty
        return False
    
    if path_parts[2] != 'reports':
        return False
    
    if not path_parts[3]:  # reportid should be non-empty
        return False
    
    # Check the query part
    query = parsed_url.query
    if query != 'experience=power-bi':
        return False
    
    return True

def is_valid_url(url):
    # Use traditional URL validation
    try:
        result = urlparse(url)
        print(f'Result is : {result}')
        return all([result.scheme, result.netloc, result.path])
    except ValueError:
        return False

# Variable initialization
report_type = 'Select'
report_url = ''
report_name = ''
extraction_url = ''
extraction_counter = False

# Constants
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(PROJECT_PATH)

# Streamlit UI
st.set_page_config(layout="wide")  # Set layout to wide

print(f'**********************************************************************************')

st.title('Report Narration')
print(f'PROJECT_PATH : {PROJECT_PATH}')

col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    report_type = st.selectbox(
        'Select Report Type', ('Select', 'PowerBI', 'Tableau', 'Cognos'))

if report_type != 'Select':
    with col2:
        report_url = st.text_input('Enter the report URL')

if report_url:
    with col3:
        report_name = st.text_input('Enter the name of the report')

if report_name:
    print(f'The report name is not empty')
    print(f'The return type of valid pbi : {is_valid_powerbi_url(report_url)}')
    print(f'The return type of url validity : {is_valid_url(report_url)}')
    print(f'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    if is_valid_powerbi_url(report_url) and is_valid_url(report_url):
        fetch = store_report_details(report_url, report_name)
        check = isextraction_required(ROOT_PATH, report_name)
        if check:
            st.write(f'Data extraction for this report is done, retype the report name `{report_name}` correctly to proceed with the Narration part')
            extraction_counter = True
        else:
            relative_notebook_path = f'data-extraction\{report_type}\{report_type}_data_extraction.ipynb'
            extraction_url = create_jupyter_link(ROOT_PATH,relative_notebook_path)
            st.markdown(f"[Open {report_type} visual data extraction file]({extraction_url})")
            extraction_counter = True

    else:
        st.error("Invalid report URL. Please enter a valid PowerBI report URL.")

# Define file path for the logo image
cg_logo_path = os.path.join(PROJECT_PATH,'Capgemini_logo.png')

cg_logo = Image.open(cg_logo_path)
if cg_logo is not None:
    st.sidebar.image(cg_logo, width=100)
#st.sidebar.image(logo, width=200)  


if extraction_counter == True:
    folder_path = os.path.join(ROOT_PATH, report_name)
    visual_data_folder = os.path.join(ROOT_PATH,report_name,"visual_data_folder")
    llm_response_folder = os.path.join(folder_path,"llm_response")
    analysis_folder = os.path.join(folder_path,"analysis")
    report_folder = os.path.join(folder_path,"reports")
    read_folder = os.path.join(folder_path,"read_folder")
    details_data_folder = os.path.join(folder_path,"details_data_folder")
    insights_folder = os.path.join(folder_path,"insights")

    # Clear visual_data_folder contents on app rerun
    try:
        if not st.session_state.get('initialized'):
            st.session_state.initialized = True
            if os.path.exists(visual_data_folder):
                shutil.rmtree(visual_data_folder)
            if os.path.exists(llm_response_folder):
                shutil.rmtree(llm_response_folder)   
            if os.path.exists(report_folder):
                shutil.rmtree(report_folder) 
            if os.path.exists(details_data_folder):
                shutil.rmtree(details_data_folder)
            if os.path.exists(analysis_folder):
                shutil.rmtree(analysis_folder)

            os.makedirs(visual_data_folder)
            os.makedirs(llm_response_folder)
            os.makedirs(report_folder)
            os.makedirs(details_data_folder)
            os.makedirs(analysis_folder)

    except:
        pass 

    print(f'''Process started....
            master folder - {folder_path}
            visual data folder - {visual_data_folder}
            llm response folder - {llm_response_folder}
            analysis folder - {analysis_folder}
            report folder - {report_folder}
        ''')

    try:
        if extraction_counter == True:
            generate_insights(read_folder,insights_folder)
            pages = ['Choose the page', 'All Pages'] + extract_pages()

            selected_page = st.sidebar.selectbox('Function name', pages, index=0)
            get_details(selected_page)

            metrics = ['Choose the metrics', 'All Metrics'] + extract_visuals()
            selected_metrics = st.sidebar.selectbox('Metrics', metrics, index=0)

            section = ['Choose an option','Full report','General analysis','Trend analysis','Impact of Covid19']
            selected_section = st.sidebar.selectbox('Section', section, index=0)

            dropdown_options = {
                'Full report': '''Provide the result in the below format : 
                        Introduction
                        Statistical analysis : Make sure you include which year has the maximum, minimum and the average values of the fields.
                        i) Maximum value
                        ii) Minimum value
                        iii) Average value
                        iv) Deviation calculation - Evaluating the deviation and how the deviation has varied over the years. Make sure that the deviation is given in terms of percentage [(Value 1 - Value 2)/Value 1] * 100
                        Summary : Give a detailed summary 
                        Conclusion : Give a firm conclusion on how the company has taken any efforts on the metrics or how much more effort is required if it is not appreciable.
                        Make sure that there are only 4 headers namely "Introduction", "Statistical analysis", "Summary" and "Conclusion" and no other headers.
                        ''',
                'General analysis': '''Provide a single paragraph response in the bulleted manner with the heading 'General analysis' and discuss the below points,
                                    The general analysis of the metrics consisting of the minimum, maximum values and average value. 
                                    Mention only the brief response having maximum values, minimum values and the average value'''
            }

            if st.sidebar.button('Generate'):
                print("**************************************************************")
                print(f"The chosen metrics is {selected_metrics}")
                print("**************************************************************")

                if selected_metrics == 'All Metrics': 
                    section_prompt = dropdown_options[selected_section]
                    get_report = report_generation(analysis_folder,section_prompt)
                else:
                    section_prompt = dropdown_options[selected_section]
                    if selected_page == 'All Pages':
                        pattern = f'*_{selected_metrics}.csv'
                        srcfile = glob.glob(os.path.join(read_folder, pattern))
                        if srcfile:
                            matching_file = srcfile[0]
                            src_visual_file = matching_file
                    else:
                        src_visual_file = os.path.join(read_folder,f'{report_name}_{selected_page}_{selected_metrics}.csv')
                        
                    src_visual_df = pd.read_csv(src_visual_file)
                    if "Unnamed: 0" in src_visual_df:
                            src_visual_df.drop(columns = "Unnamed: 0", axis = 1, inplace = True)
                    # src_visual_df.to_csv(os.path.join(visual_data_folder,f'{report_name}_{selected_page}_{selected_metrics}.csv'))
                    get_report = report_generation(analysis_folder,section_prompt)

                st.write(f"LLM triggered")
                pdf = getpdf(llm_response_folder,report_folder)
                get_report = consolidated_report(report_folder, f"{report_name}_{selected_page}_{selected_metrics}_{selected_section}.pdf")
                st.write(f"Report for {selected_page} will be generated here.")

            # Right area with tabs
            tabs = st.sidebar.radio('Navigation', ['Details', 'Narration', 'ChatBot'])

            if tabs == 'Details':
                # Add your details view here
                print(f'The details table is under progress')
                if selected_page == 'All Pages' or selected_metrics == 'All Metrics':
                    csv_files = [f for f in os.listdir(visual_data_folder) if f.endswith('.csv')]
                    # Split the CSV files into chunks for adjacent display
                    num_columns = 2  # Number of columns to display tables side-by-side
                    csv_chunks = [csv_files[i:i + num_columns] for i in range(0, len(csv_files), num_columns)]
                    
                    for chunk in csv_chunks:
                        cols = st.columns(len(chunk))
                        for idx, file in enumerate(chunk):
                            display_table = pd.read_csv(os.path.join(read_folder, file))
                            if "Unnamed: 0" in display_table.columns:
                                display_table.drop(columns="Unnamed: 0", axis=1, inplace=True)
                            
                            # Extracting page and metrics information from the file name
                            file_info = file.replace('.csv', '').split('_')
                            if len(file_info) >= 3:
                                report_name_extracted = '_'.join(file_info[:-2])
                                page_extracted = file_info[-2]
                                metrics_extracted = file_info[-1]
                                cols[idx].write(f'Displaying {metrics_extracted} visual from {page_extracted} page of {report_name_extracted} file')
                            else:
                                cols[idx].write(f'Displaying data from {file}')
                            cols[idx].write(display_table)
                else:
                    display_table = pd.read_csv(os.path.join(read_folder,f'{report_name}_{selected_page}_{selected_metrics}.csv'))
                    if "Unnamed: 0" in display_table:
                        display_table.drop(columns = "Unnamed: 0", axis = 1, inplace = True)
                    st.write(f'Displaying {selected_metrics} visual from {selected_page} page of {report_name} file')
                    st.write(display_table)

            elif tabs == 'Narration':
                if selected_page != 'Choose an option':
                    # Call the function to display the PDF
                    # report_generation(insights_folder,section_prompt)
                    display_pdf(selected_page)
                else:
                    st.write("Please select a page from the dropdown on the left to display the PDF.")

            elif tabs == 'ChatBot':
                if selected_page == 'All Pages' or selected_metrics == 'All Metrics':
                    st.write(f'Ask any questions from {selected_page} page')
                    call_csv_agent()
                else:
                    st.write(f'Ask any questions from {selected_metrics} visual of {selected_page} page')
                    ask_bot()

    except:
        pass