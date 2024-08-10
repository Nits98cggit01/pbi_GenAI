import streamlit as st
import os
import re
import urllib.parse
from urllib.parse import urlparse
import math
import shutil
from pathlib import Path
from pdf2image import convert_from_path
from PyPDF2 import PdfReader,PdfWriter
import fitz  # PyMuPDF
import base64
from PIL import Image  # Import Image from PIL
import openai
import pandas as pd
from fpdf import FPDF
from PyPDF2 import PdfMerger
import glob
from config import *

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI, AzureOpenAI
from langchain.llms import OpenAI,AzureOpenAI
from langchain_community.llms import OpenAI, AzureOpenAI


# Streamlit UI
st.set_page_config(layout="wide")  # Set layout to wide

print(f'**********************************************************************************')

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

def store_report_details(url, name):
    with open(os.path.join(ROOT_PATH, "details.txt"), 'w') as details_txt:
        details_txt.write(f'{url} \n{name}')

def is_valid_powerbi_urlz(url):
    # Define the regex for the specific PowerBI URL format
    powerbi_url_pattern = re.compile(
        r"^https://app\.powerbi\.com/groups/[a-zA-Z0-9]+/reports/[a-zA-Z0-9]+/[^?]+(\?experience=power-bi)$"
    )
    return powerbi_url_pattern.match(url) is not None

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

report_type = 'Select'
report_url = ''
report_name = ''

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
st.title('PowerBI Report Narration')
print(f'PROJECT_PATH : {PROJECT_PATH}')

ROOT_PATH = os.path.dirname(PROJECT_PATH)
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

# if report_name:
#     check = isextraction_required(ROOT_PATH, report_name)

#     if check:
#         st.write('Data extraction for this report is done')
#     else:
#         relative_notebook_path = f'data-extraction\{report_type}\{report_type}_data_extraction.ipyn'
#         extraction_url = create_jupyter_link(ROOT_PATH,relative_notebook_path)
#         st.markdown(f"[Open {report_type} visual data extraction file]({extraction_url})")
#         extraction_counter = True

if report_name:
    print(f'The report name is not empty')
    print(f'The return type of valid pbi : {is_valid_powerbi_url(report_url)}')
    print(f'The return type of url validity : {is_valid_url(report_url)}')
    print(f'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    if is_valid_powerbi_url(report_url) and is_valid_url(report_url):
        fetch = store_report_details(report_url, report_name)
        check = isextraction_required(ROOT_PATH, report_name)
        if check:
            st.write('Data extraction for this report is done')
        else:
            relative_notebook_path = f'data-extraction\{report_type}\{report_type}_data_extraction.ipyn'
            extraction_url = create_jupyter_link(ROOT_PATH,relative_notebook_path)
            st.markdown(f"[Open {report_type} visual data extraction file]({extraction_url})")
            extraction_counter = True

    else:
        st.error("Invalid report URL. Please enter a valid PowerBI report URL.")


