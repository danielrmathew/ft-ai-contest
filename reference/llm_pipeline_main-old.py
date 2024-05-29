#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import re
from datetime import datetime, timedelta
import yfinance as yf


# In[2]:


from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# In[3]:


import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate


# In[4]:


import requests
import json
from bs4 import BeautifulSoup
import os


# In[5]:


def get_llm(selected_model):
    """
    Initializes and stores LLM on memory
    selected_model: str, model name to load
    return: model variable
    """
    
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


# In[ ]:


API_TOKEN = "rxfMXxfsdsKGb8urL8shecBbBp37ZBuvU5zo39sm" # should change this to secret token
API_ENDPOINT = f"https://api.marketaux.com/v1/news/all"


# In[ ]:


payload = {
    "api_token": API_TOKEN,
    "entity_types": "equity,index,etf", # should include industries parameter
#     "industries": "Technology",
    "countries": "us",
    "published_after": "2024-05-01T13:00",
    "published_before": "2024-05-01T14:00",
    "domains": "finance.yahoo.com"
}


# In[ ]:


def get_urls(payload):
    """
    Retrieves URLs of articles from Yahoo Finance
    payload: dict, criteria for article selection
    return: list, Yahoo Finance article URLs
    """
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


# In[ ]:


def get_article_content(url):
    """
    Retrieves article content
    url: str, article URL
    return: str, article content
    """
    r = requests.get(url)
    
    if r.status_code != 200:
        return 'Invalid Response'
    
    soup = BeautifulSoup(r.content)
    body = soup.find('div', attrs={'class': 'caas-body'})
    content = ""
    
    for text in body.find_all('p'):
        content += text.text
    
    return content


# In[ ]:


def write_to_file(filename, content):
    """
    Creates and writes article content to file
    filename: str, name of file to create
    content: str, content to write to file
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)


# In[ ]:


def create_documents(start_date, num_days):
    """
    Retrieves article content over a number of days and saves the content to files in the 'articles' folder
    start_date: str, date in the format YYYY-MM-DD. articles published on this day are retrieved
    num_days: int, the number of days after start_date to retrieve articles for. i.e. if num_days=10, articles published 
        on the start_date and every day between the start_date and 10 days after are retrieved
    """
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


# In[6]:


def get_query_engine(llm):
    """
    Creates the necessary components to run a RAG pipeline
    llm: LLM object loaded into memory
    return: SimpleDirectoryReader, VectorStoreIndex, QueryEngine. the first object houses the data, the second object indexes
        the data for fast retrieval, and the third object is how we can query the data using the LLM
    """
    set_global_tokenizer(
        AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
    )
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    yf_documents = SimpleDirectoryReader(input_dir="new-articles").load_data() # meant for full_pipeline.ipynb, so in right directory
    yf_index = VectorStoreIndex.from_documents(yf_documents, embed_model=embed_model)
    yf_query_engine = yf_index.as_query_engine(llm=llm)
    
    return yf_documents, yf_index, yf_query_engine


# In[45]:


def get_stocks(query_engine, user_prefs):
    """
    Returns set of stocks to invest in based on customer preferences and market news
    llm: LLM object loaded into memory
    user_prefs: dict, has keys "num_stocks", "risk", "industries", "return_goals", contains information about user preferences
    return: list, set of stocks matching user preferences returned from LLM based on articles 
    """
    
    start_prompt = ("Gather the financial securities that are mentioned across the articles the most in a positive sentiment."
     f" The securities gathered can be stocks or ETFs. Return a list of {user_prefs['num_stocks']} securities in which no more than"
     " half of the securities are from the same industry. ")
    risk_averse = ("This list of securities should be tailored to a very risk averse investor who "
        "is more focused on stability instead of rapid growth. ")
    risk_neutral = ""
    risk_tolerant = ("This list of securities should also be tailored to a very risk tolerant investor who "
        "is more focused on rapid growth instead of stability. This list should include growth stocks with high expected returns " 
        "and the potential for greater risk. ")
    industries_prompt = f"Include multiple stocks with a strong focus on the following industry or industries: {user_prefs['industries']}. "
    end_prompt = ("Create this list so that they can be put together to make a diversified portfolio. " 
        "List the stock name, stock ticker in parentheses, and the industry associated with it " 
        "and double check that a majority of the industries aren't the same.")
    
    prompt_dict = {
        "risk_averse": risk_averse, 
        "risk_neutral": risk_neutral,
        "risk_tolerant": risk_tolerant
    }
    prompt = None
    
    if user_prefs["risk"] == "conservative":
        prompt = start_prompt + prompt_dict["risk_averse"]
    elif user_prefs["risk"] == "aggressive":
        prompt = start_prompt + prompt_dict["risk_tolerant"]
    else:
        prompt = start_prompt + prompt_dict["risk_neutral"]
        
    if user_prefs['industries']:
        prompt += industries_prompt
        
    prompt += end_prompt
        
    
    print(prompt)
    
    VALID = False
    attempts = 0
    
    while not VALID:
        attempts += 1
        print(attempts)
        response = query_engine.query(prompt)
    
        pattern = r'\(([A-Za-z]+)\)'
        stocks = re.findall(pattern, response.response)
    
    
        for ticker in stocks:
            stock = yf.Ticker(ticker)
            if 'regularMarketOpen' not in stock.info:
                VALID = False
                break
            else:
                VALID = True
                
    return stocks


# In[41]:


risk_options = ["conservative", "moderate", "aggressive"]
industry_options = ["Energy", "Materials", "Industrials", "Utilities", "Healthcare", "Financials", "Consumer Discretionary",
                    "Consumer Staples", "Technology", "Communication Services", "Real Estate"]
return_goals = []
user_prefs = {
    "num_stocks": None, 
    "risk": None, 
    "industries": None, 
}


# In[42]:


LLAMA2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"
selected_model = LLAMA2_7B_CHAT


# In[ ]:


# create_documents("2024-05-20", 3)


# In[10]:


# llm = get_llm(selected_model)


# In[11]:


# docs, index, query_engine = get_query_engine(llm)


# In[53]:


# example user preferences
user_prefs = {"num_stocks": 5, "risk": "conservative", "industries": None}

