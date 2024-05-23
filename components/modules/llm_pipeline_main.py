#!/usr/bin/env python
# coding: utf-8

# In[37]:


import time
import re
from datetime import datetime, timedelta


# In[2]:


from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# In[3]:


import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate


# In[9]:


import requests
import json
from bs4 import BeautifulSoup
import os


# In[133]:


def get_llm(selected_model):
    SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
    - Generate human readable output, avoid creating output with gibberish text.
    - Generate only the requested output, don't include any other language before or after the requested output.
    - Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
    - Generate professional language typically used in business documents in North America.
    - Never generate offensive or foul language.
    """

    query_wrapper_prompt = PromptTemplate(
        "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
    )

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=2048,
        generate_kwargs={"temperature": 0.1, "do_sample": True},
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name=selected_model,
        model_name=selected_model,
        device_map="auto",
        # change these settings below depending on your GPU
        model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
    )
        
    return llm


# In[10]:


API_TOKEN = "rxfMXxfsdsKGb8urL8shecBbBp37ZBuvU5zo39sm" # should change this to secret token
API_ENDPOINT = f"https://api.marketaux.com/v1/news/all"


# In[19]:


payload = {
    "api_token": API_TOKEN,
    "entity_types": "equity,index,etf", # should include industries parameter
#     "industries": "Technology",
    "countries": "us",
    "published_after": "2024-05-01T13:00",
    "published_before": "2024-05-01T14:00",
    "domains": "finance.yahoo.com"
}


# In[44]:


def get_urls(payload):
    r = requests.get("https://api.marketaux.com/v1/news/all", params=payload)
    output = r.text
    json_output = json.loads(output)
    data = json_output['data']
    
    urls = []
    
    for i in range(len(data)):
        url = data[i]['url']
        if url.startswith("https://finance.yahoo.com/news"):
            urls.append(url)
        
    return urls


# In[45]:


def get_article_content(url):
    r = requests.get(url)
    
    if r.status_code != 200:
        return 'Invalid Response'
    
    soup = BeautifulSoup(r.content)
    body = soup.find('div', attrs={'class': 'caas-body'})
    content = ""
    
    for text in body.find_all('p'):
        content += text.text
    
    return content


# In[46]:


def write_to_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)


# In[55]:


def create_documents(start_date, num_days):
    # start_date looks like "2024-05-01"
    payload = {
        "api_token": API_TOKEN,
        "entity_types": "equity,index,etf", 
        "countries": "us",
#         "published_after": "2024-05-10T13:00", # what time format is supposed to look like, in UTC format 
#         "published_before": "2024-05-10T14:00",
        "domains": "finance.yahoo.com"
    }
    
    def get_dates(start_date, num_days):
        # Convert the input string to a datetime object
        date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        dates = [date_obj.strftime('%Y-%m-%d')]
        # Print the input date
        print(date_obj.strftime("%Y-%m-%d"))

        # Loop to get a total of num_days days
        for i in range(1, num_days):
            next_day = date_obj + timedelta(days=i)
            dates.append(next_day.strftime('%Y-%m-%d'))
            print(f"{next_day.strftime('%Y-%m-%d')}")
        
        return dates
        
    dates = get_dates(start_date, num_days)
    
    for date in dates: 
        for add_hour in range(7): 
            start = date + "T" + str(13 + add_hour) + ":00" # edit str(xx) for start day
            finish = date + "T" + str(13 + add_hour + 1) + ":00"
            payload["published_after"] = start
            payload["published_before"] = finish
            
            urls = get_urls(payload)
            for i in range(len(urls)):
                print(urls[i])
                content = get_article_content(urls[i])
                if content != "Invalid Response": 
                    filename = os.path.join("..", "..", "articles", "article_" + start + "_" + str(add_hour) + str(i) + ".txt")
                    write_to_file(filename, content)
                else:
                    print("Above URL Skipped")


# In[128]:


def get_query_engine(llm):
    set_global_tokenizer(
        AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
    )
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    yf_documents = SimpleDirectoryReader(input_dir="articles").load_data()
    yf_index = VectorStoreIndex.from_documents(yf_documents, embed_model=embed_model)
    yf_query_engine = yf_index.as_query_engine(llm=llm)
    
    return yf_documents, yf_index, yf_query_engine


# In[132]:


def get_stocks(llm, user_prefs):
    docs, index, query_engine = get_query_engine(llm)
    
    risk_averse_prompt = "Gather the financial securities that are most mentioned across the articles in a positive sentiment." + \
    f"The securities gathered can be stocks or ETFs. Return a list of {user_prefs['num_stocks']} securities in which no more than half of the " + \
    "securities are from the same industry. This list of securities should also be tailored to a very risk averse investor who " + \
    "is more focused on stability instead of rapid growth. "
    if user_prefs['industries']:
        risk_averse_prompt += f"Include multiple stocks with a strong focus on the following industry or industries: {user_prefs['industries']}. "
    risk_averse_prompt += "Create this list so that they can be put together to make a diversified portfolio. " + \
    "List the stock name, stock ticker in parentheses, and the industry associated with it" + \
    "and double check that a majority of the industries aren't the same."
    
    risk_neutral_prompt = "Gather the financial securities that are most mentioned across the articles in a positive sentiment." + \
    f"The securities gathered can be stocks or ETFs. Return a list of {user_prefs['num_stocks']} securities in which no more than half of the " + \
    "securities are from the same industry. "
    if user_prefs['industries']:
        risk_averse_prompt += f"Include multiple stocks with a strong focus on the following industry or industries: {user_prefs['industries']}. "
    risk_averse_prompt += "Create this list so that they can be put together to make a diversified portfolio. " + \
    "List the stock name, stock ticker in parentheses, and the industry associated with it" + \
    "and double check that a majority of the industries aren't the same."
    
    risk_tolerant_prompt = "Gather the financial securities that are most mentioned across the articles in a positive sentiment." + \
    f"The securities gathered can be stocks or ETFs. Return a list of {user_prefs['num_stocks']} securities in which no more than half of the " + \
    "securities are from the same industry. This list of securities should also be tailored to a very risk tolerant investor who " + \
    "is more focused on rapid growth instead of stability. This list should include growth stocks with high expected returns " + \
    "with the potential for greater risk. Create this list so that they can be put together to make a diversified portfolio. "
    if user_prefs['industries']:
        risk_tolerant_prompt += f"Include multiple stocks with a strong focus on the following industry or industries: {industries}. " 
    risk_tolerant_prompt += "List the stock name, stock ticker in parentheses, and the industry associated with it" + \
    "and double check that a majority of the industries aren't the same."
    
    prompt_dict = {
        "risk_averse": risk_averse_prompt, 
        "risk_neutral": risk_neutral_prompt,
        "risk_tolerant": risk_tolerant_prompt
    }
    prompt = None
    
    if user_prefs["risk"] == "conservative":
        prompt = prompt_dict["risk_averse"]
    elif user_prefs["risk"] == "aggressive":
        prompt = prompt_dict["risk_tolerant"]
    else:
        prompt = prompt_dict["risk_neutral"]
    
    response = query_engine.query(prompt)
    
    pattern = r'\(([A-Za-z]+)\)'
    stocks = re.findall(pattern, response.response)
    
    return stocks


# In[99]:


risk_options = ["conservative", "moderate", "aggressive"]
industry_options = ["Energy", "Materials", "Industrials", "Utilities", "Healthcare", "Financials", "Consumer Discretionary",
                    "Consumer Staples", "Technology", "Communication Services", "Real Estate"]
return_goals = []
user_prefs = {
    "num_stocks": None, 
    "risk": None, 
    "industries": None, 
    "return_goals": None
}


# In[33]:


LLAMA2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"
selected_model = LLAMA2_7B_CHAT


# In[103]:


# example user preferences
user_prefs = {"num_stocks": 10, "risk": "conservative", "industries": None, "return_goals": None}


# In[1]:


# create_documents("2024-05-20", 3)


# In[130]:


# llm = get_llm(selected_model)
# get_stocks(llm, user_prefs)

