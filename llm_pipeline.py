#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install accelerate
# !pip install -i https://pypi.org/simple/ bitsandbytes

# !pip install llama-index-embeddings-huggingface

# !pip install llama-index

# !pip install llama-index-llms-huggingface


# In[35]:


import time
import re


# In[2]:


from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# In[3]:


import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate


# In[32]:


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


# In[13]:


import requests
import json
from bs4 import BeautifulSoup
import os


# In[14]:


API_TOKEN = "rxfMXxfsdsKGb8urL8shecBbBp37ZBuvU5zo39sm"
API_ENDPOINT = f"https://api.marketaux.com/v1/news/all"

payload = {
    "api_token": API_TOKEN,
    "entity_types": "equity,index,etf", # should include industries parameter
    "industries": "Technology",
    "countries": "us",
    "published_after": "2024-05-10T15:00",
    "domains": "finance.yahoo.com"
}


# In[15]:


payload = {
    "api_token": API_TOKEN,
    "entity_types": "equity,index,etf", # should include industries parameter
#     "industries": "Technology",
    "countries": "us",
    "published_after": "2024-05-10T13:00",
    "published_before": "2024-05-10T14:00",
    "domains": "finance.yahoo.com"
}


# In[16]:


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


# In[17]:


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


# In[18]:


def write_to_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)


# In[19]:


def create_documents():
    payload = {
        "api_token": API_TOKEN,
        "entity_types": "equity,index,etf", # should include industries parameter
    #     "industries": "Technology",
        "countries": "us",
#         "published_after": "2024-05-10T13:00",
#         "published_before": "2024-05-10T14:00",
        "domains": "finance.yahoo.com"
    }
    
    for add_day in range(3): # edit this number to decide how many days after the start day to get articles for
        for add_hour in range(7): 
            start = "2024-05-" + str(20 + add_day) + "T" + str(13 + add_hour) + ":00" # edit str(xx) for start day
            finish = "2024-05-" + str(20 + add_day) + "T" + str(13 + add_hour + 1) + ":00"
            payload["published_after"] = start
            payload["published_before"] = finish
            
            urls = get_urls(payload)
            for i in range(len(urls)):
                print(urls[i])
                content = get_article_content(urls[i])
                if content != "Invalid Response": 
                    filename = os.path.join("articles", "article" + str(add_day) + str(add_hour) + str(i) + "__.txt")
                    write_to_file(filename, content)
                else:
                    print("Above URL Skipped")


# In[20]:


# create_documents()


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


user_prefs = {"num_stocks": 10, "risk": "conservative", "industries": None, "return_goals": None}


# In[130]:


# llm = get_llm(selected_model)
# get_stocks(llm, user_prefs)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# only tech stocks except for one
# after getting new data, outputted 5 stocks with good variance


# In[26]:


# prompt = "Return the financial securities that are most mentioned across the articles in a positive sentiment." + \
# "The securities returned can be stocks or ETFs. Ensure that the securities returned are of different industries " + \
# "that can be put together to make a diversified portfolio. List the stock and the industry associated with it" + \
# "and double check that a majority of the industries aren't the same."

# response_3 = yf_query_engine.query(prompt)
# response_3.print_response_stream()


# # In[ ]:


# # this prompt only gave tech stocks excpet for one


# # In[27]:


# prompt_2 = "Gather the financial securities that are most mentioned across the articles in a positive sentiment." + \
# "The securities gathered can be stocks or ETFs. Return a list of 10 securities in which no more than half of the " + \
# "securities are from the same industry. Create this list so that they " + \
# "can be put together to make a diversified portfolio. List the stock and the industry associated with it" + \
# "and double check that a majority of the industries aren't the same."

# response_4 = yf_query_engine.query(prompt_2)
# response_4.print_response_stream()


# # In[ ]:


# # below prompt seems to provide best portfolio so far


# # In[49]:


# new_prompt_risk_averse = "Gather the financial securities that are most mentioned across the articles in a positive sentiment." + \
# "The securities gathered can be stocks or ETFs. Return a list of 10 securities in which no more than half of the " + \
# "securities are from the same industry. This list of securities should also be tailored to a very risk averse investor who " + \
# "is more focused on stability instead of rapid growth. " + \
# "Create this list so that they " + \
# "can be put together to make a diversified portfolio. List the stock name, stock ticker in parentheses, and the industry associated with it" + \
# "and double check that a majority of the industries aren't the same."

# response_risk = yf_query_engine.query(new_prompt_risk_averse)
# response_risk.print_response_stream()


# # In[58]:


# new_prompt_risk_tolerant = "Gather the financial securities that are most mentioned across the articles in a positive sentiment." + \
# "The securities gathered can be stocks or ETFs. Return a list of 10 securities in which no more than half of the " + \
# "securities are from the same industry. This list of securities should also be tailored to a very risk tolerant investor who " + \
# "is more focused on rapid growth instead of stability. " + \
# "Create this list so that they " + \
# "can be put together to make a diversified portfolio. List the stock name, stock ticker in parentheses, and the industry associated with it" + \
# "and double check that a majority of the industries aren't the same."

# response_no_risk = yf_query_engine.query(new_prompt_risk_tolerant)
# response_no_risk.print_response_stream()


# # In[82]:


# industries = ["industrials", "communication services", "real estate"]


# # In[83]:


# new_prompt_risk_tolerant = "Gather the financial securities that are most mentioned across the articles in a positive sentiment." + \
# "The securities gathered can be stocks or ETFs. Return a list of 10 securities in which no more than half of the " + \
# "securities are from the same industry. This list of securities should also be tailored to a very risk tolerant investor who " + \
# "is more focused on rapid growth instead of stability. This list should include growth stocks with high expected returns " + \
# "with the potential for greater risk. Create this list so that they " + \
# "can be put together to make a diversified portfolio. Include multiple stocks with a strong focus on the " + \
# f"following industry or industries: {industries}. " + \
# "List the stock name, stock ticker in parentheses, and the industry associated with it" + \
# "and double check that a majority of the industries aren't the same."

# response_no_risk = yf_query_engine.query(new_prompt_risk_tolerant)
# response_no_risk.print_response_stream()


# # In[56]:


# prompts_dict = yf_query_engine.get_prompts()
# print(list(prompts_dict.keys()))


# # In[60]:


# from llama_index.core import PromptTemplate


# # In[65]:


# text_qa_template_str = (
#     "Context information is"
#     " below.\n---------------------\n{context_str}\n---------------------\nUsing"
#     " both the context information and also using your own knowledge, answer"
#     " the request about portfolio recommendations: {query_str}\nIf the context isn't helpful, you can also"
#     " answer the request on your own.\nWhen answering,"
#     " respond in the following format:\nSTOCK NAME (STOCK TICKER) - INDUSTRY\n and"
#     " follow that format for each stock in the portfolio. The words in all capitals should be replaced with"
#     " the actual stock name, the stock ticker, and industry, respectively. Only include the stock recommendations,"
#     " do not include any filler text or notes at the beginning or end of the response."
# )
# text_qa_template = PromptTemplate(text_qa_template_str)


# # In[66]:


# # should test adding system prompt and incorporating user preferences like risk aversion and industry preferences


# # In[67]:


# yf_query_engine_prompt = yf_index.as_query_engine(llm=llm, streaming=True, text_qa_template=text_qa_template)


# # In[64]:


# response_risk_prompt = yf_query_engine.query(new_prompt_risk_averse)
# response_risk_prompt.print_response_stream()


# # In[68]:


# response_no_risk_prompt = yf_query_engine_prompt.query(new_prompt_risk_tolerant)
# response_no_risk_prompt.print_response_stream()


# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[29]:


# from dual_dcf_model import discounted_cash_flow

