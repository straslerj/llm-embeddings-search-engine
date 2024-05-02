import pandas as pd
from openai import OpenAI
import csv
import time


OPENAI_KEY = "XXXXXXXX" # I used mine but wanted to hide it for posting on git
system_input = "<|SYSTEM|>\nYou are a professor teaching a course. You are knowledgeable of the syllabus.\n<|USER|>\nPlease answer the following question using the context provided. If you don't know the answer, just say that you don't know.\nBase your answer on the context below. Say 'I don't know' if the answer does not appear to be in the context below."


def llm_return(client, selected_model, system_input, user_input, temperature = 0, top_p = 0.95, max_tokens = 4000):
        feed_back = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_input},
                    {"role": 'user', "content": user_input}]
                ,
                temperature=temperature,
                top_p=top_p,
                max_tokens = max_tokens
                )
        return feed_back
    
syllabus = pd.read_csv("syllabus_responses.csv")
client = OpenAI(api_key=OPENAI_KEY)
contexts = syllabus[['question','context']].drop_duplicates()

result_df = pd.DataFrame(columns = ['question','context','llm','llm_response','time_taken'])

for i in range(len(contexts)):
    context = contexts.iat[i,1]
    question = contexts.iat[i,0]
    user_template = f"QUESTION: {question} \nCONTEXT: \n{context}"
    start_time = time.time()
    response = llm_return(client,"gpt-3.5-turbo",system_input,user_template)
    end_time = time.time()
    result_df.loc[len(result_df.index)] = [question, context, 'gpt3.5-turbo',response.choices[0].message.content,end_time - start_time]
    
    
store = pd.concat([syllabus,result_df],axis = 0)
store.to_csv("syllabus_responses.csv")

print(result_df['time_taken'].mean())
