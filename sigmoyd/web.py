from langchain_google_genai import ChatGoogleGenerativeAI
import bs4
import requests as re
import json
import selenium
import os

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=GEMINI_API_KEY)

class WEB:
    
    def __init__(self):
        self.llm = llm
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
        
        
        































