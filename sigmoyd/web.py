import aiohttp
import asyncio
import os
import json
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ddgs import DDGS
from sentence_transformers import SentenceTransformer, util

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Model and Environment Initialization ---
# Load environment variables at the start
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# Initialize models outside the class as requested, so they are loaded only once.
print("Initializing models...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Parser LLM for structured data extraction (low temperature)
parser_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
    temperature=0.0,
)

# Main LLM for synthesis and creative tasks (higher temperature)
main_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
    temperature=0.7,
)
print("Models initialized.")


class WEB_SEARCH:
    """
    An AI-powered web search agent that takes a natural language query,
    intelligently parses it, searches the web, scrapes and ranks content,
    and synthesizes a comprehensive answer.
    """

    def __init__(self):
        """
        Initializes the WEB_SEARCH agent.
        Models are loaded in the global scope to be instantiated only once.
        """
        pass

    def _parse_user_query(self, user_query: str) -> dict:
        """
        Uses a lightweight LLM to parse the user's natural language query into a
        structured dictionary containing a search query and an output format.

        Args:
            user_query (str): The raw query from the user.

        Returns:
            dict: A dictionary with 'search_query' and 'output_format'.
        """
        print("1. Parsing user query...")
        
        
        prompt_text = """
        You are a highly efficient query parsing assistant. Your task is to analyze the user's request and break it down into a concise "search_query" for a web search engine and an "output_format" instruction for another AI.

        Provide your response ONLY in the form of a valid JSON object. Do not add any explanatory text before or after the JSON. Your entire response should be the JSON object itself.

        Here are some examples:

        User Request: "Find out about the latest advancements in AI and give me a 3-point summary."
        {{"search_query": "latest advancements in artificial intelligence", "output_format": "a 3-point summary"}}

        User Request: "Write a blog post about the benefits of remote work for small businesses."
        {{"search_query": "benefits of remote work for small businesses", "output_format": "a blog post"}}

        User Request: "What is the capital of Mongolia?"
        {{"search_query": "capital of Mongolia", "output_format": "a concise answer"}}

        Now, parse the following user request:

        User Request: "{user_query}"
        """


        parser_prompt = PromptTemplate(
            template=prompt_text,
            input_variables=['user_query']
        )
        
        # Uses the global parser_llm
        llm_chain = parser_prompt | parser_llm
        
        try:
            llm_response = llm_chain.invoke({'user_query': user_query})
            cleaned_response = llm_response.content.strip().replace('```json', '').replace('```', '').strip()
            parsed_json = json.loads(cleaned_response)
            print(f"   - Successfully parsed query: {parsed_json}")
            return parsed_json
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            print(f"   - Warning: Failed to parse LLM response ({e}). Using default.")
            return {
                'search_query': user_query,
                'output_format': 'a concise summary'
            }

    def _fetch_search_results(self, search_query: str, max_results: int = 20) -> list:
        """
        Fetches web search results using DuckDuckGo.

        Args:
            search_query (str): The query to search for.
            max_results (int): The maximum number of results to fetch.

        Returns:
            list: A list of search result dictionaries.
        """
        print(f"2. Fetching top {max_results} web results for: '{search_query}'...")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=max_results))
            print(f"   - Found {len(results)} results.")
            return results
        except Exception as e:
            print(f"   - Error fetching search results: {e}")
            return []

    def _rank_results_by_similarity(self, search_query: str, results: list) -> list:
        """
        Ranks search results based on the cosine similarity between the query
        and the result titles.

        Args:
            search_query (str): The original search query.
            results (list): The list of search results from DuckDuckGo.

        Returns:
            list: A sorted list of results with an added 'similarity' score.
        """
        print("3. Ranking results by semantic similarity...")
        if not results:
            return []
            
        query_embedding = embedding_model.encode(search_query, convert_to_tensor=True)
        result_titles = [result.get('title', '') for result in results]
        
        title_embeddings = embedding_model.encode(result_titles, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(query_embedding, title_embeddings)
        
        for i, result in enumerate(results):
            result['similarity'] = cosine_scores[0][i].item()
            
        ranked_results = sorted(results, key=lambda x: x.get('similarity', 0), reverse=True)
        print("   - Ranking complete.")
        return ranked_results
        
    async def _scrape_websites_content_async(self, urls: list) -> tuple:
        """
        Scrapes the main text content from a list of URLs concurrently, limited to 200 words each.

        Args:
            urls (list): List of URLs to scrape.

        Returns:
            tuple: (context_parts, sources_used)
                   context_parts is a list of extracted text content strings,
                   sources_used is a list of URLs that were successfully scraped.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        async def fetch_and_parse(session, url):
            """Fetch and parse a single URL"""
            try:
                async with session.get(url, timeout=5) as response:
                    response.raise_for_status()
                    # Read content with size limit
                    content = await response.read()
                    if len(content) > 500 * 1024:  # Limit to 500KB
                        content = content[:500 * 1024]

                # Parse with BeautifulSoup and lxml
                soup = BeautifulSoup(content, 'lxml')
                
                # Remove non-visible elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
                    element.decompose()

                # Extract text
                text_chunks = [p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'article', 'div'])]
                full_text = ' '.join(text_chunks).strip()
                cleaned_text = ' '.join(full_text.split())

                # Truncate to 200 words
                words = cleaned_text.split()
                if len(words) > 200:
                    extracted_text = ' '.join(words[:200]) + "..."
                else:
                    extracted_text = ' '.join(words)

                return url, extracted_text

            except (aiohttp.ClientError, asyncio.TimeoutError, UnicodeDecodeError, AttributeError) as e:
                print(f"   - Failed to scrape {url}: {str(e)}")
                return url, ""

        context_parts = []
        sources_used = []

        # Create aiohttp session and fetch all URLs concurrently
        timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            # Create tasks for all URLs
            tasks = [fetch_and_parse(session, url) for url in urls]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                continue
            
            url, content = result
            if content:  # Only add if content was successfully extracted
                context_parts.append(f"Content from {url}:\n{content}")
                sources_used.append(url)

        return context_parts, sources_used

    def _invoke_llm(self, original_query: str, context: str, output_format: str) -> str:
        """
        Invokes the main LLM to synthesize an answer based on the provided context.

        Args:
            original_query (str): The user's original, full query.
            context (str): The aggregated content scraped from websites.
            output_format (str): The desired format for the output.

        Returns:
            str: The final, synthesized response from the LLM.
        """
        print("5. Synthesizing final answer with main LLM...")
        
        prompt_text = """
        You are an expert research assistant. Your task is to provide a comprehensive and well-structured answer to the user's query based *only* on the provided context from web search results. Do not use any prior knowledge.

        Here is the user's original query:
        "{original_query}"

        Here is the context scraped from the web:
        ---
        {context}
        ---

        Based on the context above, please generate a response that fulfills the user's request.
        The desired output format is: "{output_format}".
        
        Ensure your answer is accurate, coherent, and directly addresses the user's question using only the information given in the context.
        """

        main_prompt = PromptTemplate(
            template=prompt_text,
            input_variables=['original_query', 'context', 'output_format']
        )

        # Uses the global main_llm
        main_chain = main_prompt | main_llm
        
        response = main_chain.invoke({
            'original_query': original_query,
            'context': context,
            'output_format': output_format
        })
        
        print("   - Synthesis complete.")
        return response.content

    def ALL_Action(self, user_query: str) -> dict:
        """
        The main public method that orchestrates the entire process from
        query parsing to final answer generation.

        Args:
            user_query (str): The user's natural language query.

        Returns:
            dict: A structured dictionary containing the final response and metadata.
        """
        total_start_time = time.time()
        timings = {}

        # 1. Parse the user query
        step_start_time = time.time()
        parsed_query_dict = self._parse_user_query(user_query)
        timings['1_parse_query'] = time.time() - step_start_time

        if not parsed_query_dict or 'search_query' not in parsed_query_dict:
            return {
                "error": "Failed to parse the user query.",
                "final_response": None,
                "sources_used": []
            }
        search_query = parsed_query_dict['search_query']
        output_format = parsed_query_dict['output_format']

        # 2. Fetch search results
        step_start_time = time.time()
        results = self._fetch_search_results(search_query)
        timings['2_fetch_results'] = time.time() - step_start_time

        # 3. Rank results
        step_start_time = time.time()
        ranked_results = self._rank_results_by_similarity(search_query, results)
        timings['3_rank_results'] = time.time() - step_start_time

        # 4. Scrape top N results
        step_start_time = time.time()
        print("4. Scraping content from top 3 ranked websites...")
        top_n_to_scrape = 5
        urls_to_scrape = [
            result.get('href') for result in ranked_results[:top_n_to_scrape] if result.get('href')
        ]
        
        context_parts, sources_used = asyncio.run(self._scrape_websites_content_async(urls_to_scrape))
        
        timings['4_scrape_content'] = time.time() - step_start_time
        context = "\n\n---\n\n".join(context_parts)
        
        if not context.strip():
            print("   - Error: No content could be scraped from the top results.")
            return {
                "error": "Could not retrieve content from any of the top search results.",
                "final_response": None,
                "sources_used": [],
                "parsed_query": parsed_query_dict
            }

        # 5. Generate the final response
        step_start_time = time.time()
        final_response = self._invoke_llm(user_query, context, output_format)
        timings['5_synthesize_answer'] = time.time() - step_start_time
        
        timings['total_execution_time'] = time.time() - total_start_time
        
        print("\n--- Performance Timings ---")
        for step, duration in timings.items():
            print(f"- {step.replace('_', ' ').capitalize()}: {duration:.2f} seconds")
        print("---------------------------\n")

        # 6. Structure and return the final output
        final_output = {
            "user_query": user_query,
            "parsed_query": parsed_query_dict,
            "final_response": final_response,
            "sources_used": sources_used,
            "performance_timings_seconds": {k: round(v, 2) for k, v in timings.items()},
            "top_ranked_results_for_review": ranked_results[:top_n_to_scrape]
        }
        
        print("\nProcess finished successfully!")
        return final_output

# --- Example Usage ---
if __name__ == "__main__":
    # Initialize the agent
    search_agent = WEB_SEARCH()

    # --- Test Queries ---
    # Query 1: Simple fact-finding
    # user_input = "What were the key findings of the latest IPCC report on climate change?"
    
    # Query 2: Creative, formatted output
    # user_input = "Write a short blog post about the benefits of using Python for data science, aimed at beginners."
    # user_input = "write a blog on ai agents with keywords optimized for SEO"
    user_input = "give me judgments on the authority of court to pass order that is preventive in nature and preserve status quo ante"
    
    # Query 3: Summarization
    # user_input = "What are the latest developments in quantum computing? Give me a 3-point summary."

    print(f"\n--- Starting Web Search Agent for query: '{user_input}' ---\n")
    
    # Run the agent
    final_result = search_agent.ALL_Action(user_input)

    print("\n--- Final Agent Output ---\n")
    # Pretty-print the JSON output
    print(json.dumps(final_result, indent=2))
    print("\n--- End of Report ---\n")

