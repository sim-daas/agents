from composio_langchain import ComposioToolSet, Action, App
from auth import create_connection_oauth2 , check_connection
from langchain_google_genai import ChatGoogleGenerativeAI
import composio_langchain
import json
import re
import ast
import sqlite3
import pandas as pd
import operator

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key="")  

class GOOGLESHEETS:
    
    def __init__(self,api_key,kwargs):
        self.api_key = api_key
        self.prompt_toolset = composio_langchain.ComposioToolSet(api_key=api_key)
   
    def clean_llm_json_response(self,raw):
        # Strip language markers like ```json or '''json
        cleaned = re.sub(r"^(```|''')json", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"(```|''')$", "", cleaned.strip())
        return json.loads(cleaned)
    
    def clean_sql_query(self,query):
        cleaned_query = re.sub(r'```sql|```', '', query).strip()

        # Step 2: Remove extra spaces around SQL keywords and operators.
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()

        # Step 3: Remove unnecessary quotes around string values (if any).
        cleaned_query = re.sub(r'"([^"]+)"', r'\1', cleaned_query)

        # Step 4: Remove any backticks around table or column names
        cleaned_query = re.sub(r'`([^`]+)`', r'\1', cleaned_query)

        # Step 5: Clean column names with special characters like `PHONE-NO.`
        cleaned_query = re.sub(r'([A-Za-z0-9])\s*-\s*([A-Za-z0-9])', r'\1_\2', cleaned_query)

        return cleaned_query

    def GOOGLESHEETS_BATCH_GET(self,spreadsheetID):
        
        params={'spreadsheet_id':spreadsheetID}
        composio_toolset = ComposioToolSet(api_key=self.api_key)
        response=composio_toolset.execute_action(params=params,action = 'GOOGLESHEETS_BATCH_GET' )
        
        header=response['data']['valueRanges']['values'][0]
        rows = response['data']['valueRanges']['values'][1:]
           
        mapped_data = [dict(zip(header, row)) for row in rows]
        return {'response':mapped_data}
    def get_sheet_header(self,spreadsheetID):
        print("")
        response=self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID)
        
        headers=[]
       
        for key in response['response'][0].keys():
            headers.append(key)
        return headers
    def header_description(self,spreadsheetID):
        
        header=self.get_sheet_header(spreadsheetID=spreadsheetID)
        data=self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID)
        
        description={}
        for j in header:
            
            col_data=[]
            for i in data['response']:
                col_data.append(i[j])

            task =f'''
                {j} is a column name from a table , which contains a data as {col_data} . now understand the pattern of the data and give a brief description of the data and its usage in the table.
                this data can be any entity so while making the description keep that in the mind . give the description in a strict way and do not give any reasoning or explanation or approach.
            STRICTLY FOLLOW THE ABOVE TASK GIVEN AND EXTRACT INFORMATION FROM THE QUERY CONTENT NO REASONING NO EXPLANATION NO APPROCH IS REQUIRED JUST THE DATA '''
                    
            response1 = llm.invoke(task)
            description[j]=response1.content.strip().lower()
        
        return description
            # print(j," :- ",response1.content.strip().lower())
    def add_new_data(self,spreadsheetID,data:list):
        response=self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID)
        composio_toolset = ComposioToolSet(api_key=self.api_key)
        new_row=len(response['response'])+2
        print(new_row)
        response1=composio_toolset.execute_action(params={'spreadsheet_id':spreadsheetID},action = 'GOOGLESHEETS_BATCH_GET' )
        sheet_name=response1['data']['spreadsheet_data']['valueRanges']['range'].split('!')[0]
        params={'spreadsheet_id':spreadsheetID,'sheet_name':sheet_name,'first_cell_location':f'A{new_row}','values':[data],'includeValuesInResponse':False,'valueInputOption':'USER_ENTERED'}
        composio_toolset = ComposioToolSet(api_key=self.api_key)
        response3=composio_toolset.execute_action(params=params,action = 'GOOGLESHEETS_BATCH_UPDATE' )
        return {'response':response3['successfull']} 
    def update_cell(self,spreadsheetID,data:str,column_name,old_value):
        response=self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID)
        #print(response['response'][1])
        k=1
        for i in response['response']:
            #print(i)
            if i[column_name]==old_value:
                row=k
                header=self.get_sheet_header(spreadsheetID=spreadsheetID)
                column_number=header.index(column_name)+1
                column_letter=chr(64+column_number)
                cell_location=f'{column_letter}{row+1}'
                params={'spreadsheet_id':spreadsheetID,'sheet_name':'Sheet1','first_cell_location':cell_location,'values':[[data]],'includeValuesInResponse':False,'valueInputOption':'USER_ENTERED'}
                try:
                    composio_toolset = ComposioToolSet(api_key=self.api_key)
                    response=composio_toolset.execute_action(params=params,action = 'GOOGLESHEETS_BATCH_UPDATE' )
                    return {'response':response['successfull']}
                except Exception as e:
                    return {'error':str(e)}
            else:
                k+=1
    def update_cell_by_row_condition(self,spreadsheetID,data:str,column_name,condition:dict):
        response=self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID)
        #print(response['response'][1])
        k=1
        for i in response['response']:
            #print(i)
            if all(item in i.items() for item in condition.items()):
                row=k
                header=self.get_sheet_header(spreadsheetID=spreadsheetID)
                column_number=header.index(column_name)+1
                column_letter=chr(64+column_number)
                cell_location=f'{column_letter}{row+1}'
                params={'spreadsheet_id':spreadsheetID,'sheet_name':'Sheet1','first_cell_location':cell_location,'values':[[data]],'includeValuesInResponse':False,'valueInputOption':'USER_ENTERED'}
                try:
                    composio_toolset = ComposioToolSet(api_key=self.api_key)
                    response=composio_toolset.execute_action(params=params,action = 'GOOGLESHEETS_BATCH_UPDATE' )
                    return {'response':response['successfull']}
                except Exception as e:
                    return {'error':str(e)}
            else:
                k+=1
    def update_condition_divider(self,spreadsheetID,query:str):
        details=self.header_description(spreadsheetID=spreadsheetID)
        header=self.get_sheet_header(spreadsheetID=spreadsheetID)

        task1=f'''for the given query : {query} there are two types of data present one is the data which is to be updated and other is the condition on which the data is to be updated.
        now find the data which is to be updated , which means the new data which will replace the older one and return just the data.
        the final response will be containing a single entity as the data which is to be updated.
        
        STRICTLY FOLLOW THE ABOVE TASK GIVEN AND EXTRACT INFORMATION FROM THE QUERY CONTENT NO REASONING NO EXPLANATION NO APPROCH IS REQUIRED JUST THE DATA '''
        response1=llm.invoke(task1)
        data= response1.content.strip()
        print(data)
        task2=f'''from the given query :{query} it has been understood that we need to update some value in a table with {data} but yet don't know in which column . 
         now look into column_detail where every key is the column name and the value is the detail of the column : {details} and find out the in which column the updation to {data} is required 
         the final response will be containing a single entity as the column name where we have to update. {header} these are the column name so the entity should be from this list. 
        STRICTLY FOLLOW THE ABOVE TASK GIVEN AND EXTRACT INFORMATION FROM THE QUERY CONTENT NO REASONING NO EXPLANATION NO APPROCH IS REQUIRED JUST THE DATA'''
        response2=llm.invoke(task2)
        column_name= response2.content.strip()
        print(column_name)

        task3=f'''from the query : {query} it has been understood that we need to upade at column {column_name} with {data} byt yet we don't know that in which row .
        the row requirement has been in the query . now we have to find out the row condition then follow these steps:
        step 1 : in query:{query} there must be some value which will be acting as the condition for the row and its not {data}
        step 2 : once value has been found look into these column details where every every key is taken from {header} and every value depicts the column description : {details} 
        step 3 : make the dictonary of the column name found in the step 2 and value in step 1 in the format : 'column_name':value . also there can be be more than one conditional columns
        
        the final response must contain a single entity as the json of format 
        column name : value
        NO PRIEMBLES AND POSTAMBLES ARE REQUIRED , AND KEEP ALL STRINGS IN DOUBLE QUOTES'''
        response3=llm.invoke(task3)
        print("bhej dia hai guru")
        print(response3.content.strip())
        condition=self.clean_llm_json_response(response3.content.strip())
        print(condition)
        print(type(condition))

        return {'data':data,'column_name':column_name,'condition':condition}
    def standardize_methodology_update(self,spreadsheetID,query:str):
        
        df=pd.DataFrame(self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID)['response'])
        df.columns = [re.sub(r'\W+', '', col).lower() for col in df.columns]
        
            

        # Ask LLM to return structured update instruction
        task = f'''
        Given this user query: "{query}", return a JSON object describing the update.
        
        Available columns: {df.columns.tolist()}
        Column descriptions: {self.header_description(spreadsheetID=spreadsheetID)}
        Column data types: {df.dtypes.astype(str).to_dict()}
        
        Return only this format:
        {{
        "column": "column_to_update",
        "value": "new_value",
        "condition": {{
            "column": "condition_column",
            "value": condition_value
        }}
        }}

        - Use string values in double quotes.
        - Use correct Python types (e.g. strings in quotes, numbers as-is).
        - Do not include any preambles and postembles .
        '''

        response = llm.invoke(task).content.strip()
       
        update_instruction=self.clean_llm_json_response(response)
        print(update_instruction)
        # Extract and apply the update
        col_to_update = update_instruction['column']
        new_value = update_instruction['value']
        condition = update_instruction['condition']
        cond_col = condition['column']
        cond_val = condition['value']

        # Apply the update
        try:
            df.loc[df[cond_col] == cond_val, col_to_update] = new_value
            data=[]
            for i in df.values.tolist():
                data.append(list(i))

            params={'spreadsheet_id':spreadsheetID,'sheet_name':'Sheet1','first_cell_location':'A2','values':data,'includeValuesInResponse':False,'valueInputOption':'USER_ENTERED'}
            try:
                composio_toolset = ComposioToolSet(api_key=self.api_key)
                response=composio_toolset.execute_action(params=params,action = 'GOOGLESHEETS_BATCH_UPDATE' )
                return {'response':response['successfull']}
            except Exception as e:
                return {'error':str(e)}
        except KeyError as e:
            return f"Error: {e}. Column '{col_to_update}' or '{cond_col}' not found in DataFrame."
          # Debug print

        # Optional: update Google Sheet back here with updated df
    def standardize_methedology_read(self , spreadsheetID , query:str):
        
        # Load and clean sheet data into a DataFrame
        #print(self.header_description(spreadsheetID=spreadsheetID))
        df = pd.DataFrame(self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID)['response'])
        df.columns = [re.sub(r'\W+', '', col).lower() for col in df.columns]
        #print(df.dtypes)

        # Construct LLM prompt
        task = f'''
            Given this user query: "{query}", return a JSON object describing the read operation on a pandas DataFrame.

            Available columns: {df.columns.tolist()}
            Column descriptions: {self.header_description(spreadsheetID=spreadsheetID)}
            Column data types: {df.dtypes.astype(str).to_dict()}

            Return only this format:
            {{
            "select_columns": ["col1", "col2", ...],
            "condition": {{
                "column": "condition_column",
                "operator": "comparison_operator",   # e.g., ">", "<", "==", ">=", "<=", "!="
                "value": condition_value
            }}
            }}
            the above made format is basically used for reading specific data , but in case of read complete data the above condition will have all null . Or in case of reading a column the operator and the value will be null. 

            - Use string values in double quotes.
            - Use correct Python types (e.g. strings in quotes, numbers as-is).
            - Do not include any preambles or postambles.
            '''


        ops = {
            "==": operator.eq,
            "!=": operator.ne,
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le
        }
        spec=self.clean_llm_json_response(llm.invoke(task).content.strip())
        #print(spec)
        select_cols = spec["select_columns"]
        cond = spec["condition"]
        
        if cond['operator']==None or "None":
            
            if cond['column']==None or "None":

                if cond['value']==None or "None":
                    return df.to_dict(orient='records')
            
            else:
                return df[select_cols].to_dict(orient='records')
        
        else:
            cond_col = cond["column"]
            cond_op = cond["operator"]
            cond_val = cond["value"]

            # Safe results list
            results = []

            # Loop through rows
            for _, row in df.iterrows():
                cell_val = row[cond_col]

                try:
                    # Convert types for safe comparison
                    if isinstance(cond_val, (int, float)):
                        cell_val = float(cell_val)
                    elif isinstance(cond_val, bool):
                        cell_val = str(cell_val).strip().lower() in ['true', '1', 'yes']
                    else:
                        cell_val = str(cell_val)

                    if ops[cond_op](cell_val, cond_val):
                        results.append({col: row[col] for col in select_cols})
                        return results
                except Exception as e:
                    print(f"Skipping row due to error: {e}")
                    continue

            # Output
            print(results)
    def standardize_methodology_add(self,spreadsheetID,query:str):
        print("finding sheet header")
        header=self.get_sheet_header(spreadsheetID=spreadsheetID)
        print("finding sheet header description")
        header_description=self.header_description(spreadsheetID=spreadsheetID)
        task= f'''you are a query to data maker agent that looks into the {query} and try to find the appropriate data that suits the {header}
                the headers : {header} has descriptions as {header_description}. 
                expected output is a json with -
                1. keys as {header}
                2. values corresponding to the {header} that you have extracted from {query} 
                3. if for any in {header} no appropriate value is found then make its value as None
                no preambles , postambles and explainantion needed just the json'''
        response=llm.invoke(task)
        print("getting LLM response")
        data=self.clean_llm_json_response(response.content)
        print(data)
        rows=self.GOOGLESHEETS_BATCH_GET(spreadsheetID=spreadsheetID)['response']
        print(rows)
        new_data_location='A'+str(len(rows)+2)
        print(new_data_location)
        data_list = ['' if v is None else str(v) for v in data.values()]
        print(data_list)
        params={'spreadsheet_id':spreadsheetID,'sheet_name':'Sheet1','first_cell_location':new_data_location,'values':[data_list],'includeValuesInResponse':False,'valueInputOption':'USER_ENTERED'}
        try:
            composio_toolset = ComposioToolSet(api_key=self.api_key)
            response=composio_toolset.execute_action(params=params,action ='GOOGLESHEETS_BATCH_UPDATE')
            print('executed')
            return {'response':response['successfull']}
        except Exception as e:
            return {'error':str(e)}
    
    def standardize_methodology_create(self,query:str):
        """
        Processes a natural language query to extract structured data entries and creates a Google Sheet from them.
        Args:
            query (str): A natural language string containing data entries to be extracted and tabulated.
        Returns:
            dict: 
                - On success: {'sheet_url': <URL of the created Google Sheet>}
                - On failure: {'error': <error message>}
        
        """

        task = f'''You are a Google Sheets agent. Given the following query: "{query}", extract all data entries that should become rows in a sheet.

    For each entry in the query, create a JSON object where:
    - The keys are the column names (as inferred from the query, e.g., "name", "product", "quantity", etc.).
    - The values are the corresponding values extracted from the query.

    If the query contains multiple data entries, return a list of JSON objects (one per row). 
    If a value for a column is missing in an entry, set its value to null.

    Return only the list of JSON objects, no explanation or extra text. Example:
    [
      {{"name": "Alice", "product": "chips", "quantity": 3, "amount": 60}},
      {{"name": "Bob", "product": "soda", "quantity": 2, "amount": 40}}
    ]
    '''
        json_response = llm.invoke(task).content
        print(json_response)
        try:
            list_composio=self.clean_llm_json_response(json_response)
            print("cleaned response :",list_composio ,type(list_composio))
            try:
                task_new=f'''you are a google sheets agent that looks into the {list_composio} and try to find the appropriate title that suits the google sheet 
                expected output is just a small string which will be title that suits the google sheet and no preambles , postambles and explainantion needed just the string'''
                title=llm.invoke(task_new).content
                params={'title':title,'sheet_name':'Sheet1','sheet_json':list_composio}
                composio_toolset = ComposioToolSet(api_key=self.api_key)
                response3=composio_toolset.execute_action(params=params,action = 'GOOGLESHEETS_SHEET_FROM_JSON')
                #https://docs.google.com/spreadsheets/d/1JbboZxp5tzVSEWSCcbZllH288dcgj5xy7c1ciSuOlpE
                return {'sheet_url':"https://docs.google.com/spreadsheets/d/"+str(response3['data']['response_data']['spreadsheetId'])}
            except Exception as e:
                print("Error creating sheet:", e)
                return {'error': str(e)}
        except Exception as e:
            print("Error cleaning response:", e)

    def update(self,spreadsheetID,query: str = 'this may contain request for updation of any data, creation of any sheet and reading any specific data'):
        task = f"""
                    You are an AI assistant classifying actions for Google Sheets operations.
                    Given the following query:
                    "{query}"
                    Classify whether the intent is to 'create', 'update', 'read' or 'add a new row or column' the data.
                    Respond with only one word: create,update,read,add.
                    """
        # Call the model
        response = llm.invoke(task)

        # If using Langchain's return object
        classifies=response.content.strip().lower()

        print(classifies)
        if classifies=='update':
            try:    
                response=self.standardize_methodology_update(spreadsheetID=spreadsheetID,query=query)
                print(response)
            except Exception as e:
                return {'error':str(e)}
            # If using Langchain's return object
                # print(col_data)
        elif classifies=='read':
            try:
                response=self.standardize_methedology_read(spreadsheetID=spreadsheetID,query=query)
                print(response)
            except Exception as e:
                return {'error':str(e)}
        elif classifies=='add':
            try:
                response=self.standardize_methodology_add(spreadsheetID=spreadsheetID,query=query)
        
            
            except Exception as e:
                return {'error':str(e)}
        elif classifies=='create':
            try:
                response=self.standardize_methodology_create(query=query)
                print(response)
            except Exception as e:
                return {'error':str(e)}
            
        

            


                

        




    
# #https://docs.google.com/spreadsheets/d/1Y7lWXg0c3JzgkNa1OS2WqC4C7NbloHLKBvHHJf1pzfw/edit?usp=sharing
# try:
#     if check_connection(app_name="GOOGLESHEETS",api_key="wfaixhni71caogru03zu7a")==True:
#         sheet=GOOGLESHEETS(api_key="wfaixhni71caogru03zu7a",kwargs={})
# #response=sheet.GOOGLESHEETS_BATCH_GET(spreadsheetID="1Y7lWXg0c3JzgkNa1OS2WqC4C7NbloHLKBvHHJf1pzfw")
#         print("hello ji kre start ?")
#         response=sheet.standardize_methodology_create(query='''  
#   "task1": "Meeting with project team from 10 to 11 am",
#   "task2": "Dentist appointment from 12 to 12:30 pm",
#   "task3": "Webinar on AI at 6 pm"''')
#         print(response)
#     else:
#         create_connection_oauth2(app_name="GOOGLESHEETS",api_key="wfaixhni71caogru03zu7a", auth_scheme="OAUTH2")
# except:
#     print(create_connection_oauth2(app_name="GOOGLESHEETS",api_key="wfaixhni71caogru03zu7a", auth_scheme="OAUTH2"))

j={'insight': 'Female customers generate a higher total sales amount compared to male customers.', 'representation': 'bar graph', 'params': {'labels': '<array of categories (e.g., product types)>', 'data': '<array of values corresponding to each category>'}}
values_list = list(j.values())
print(values_list)