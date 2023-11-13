import json
import keys
from openai import OpenAI
import numpy as np
import pandas as pd
import tiktoken
from termcolor import colored


client = OpenAI(api_key=keys.OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(question):
    response = client.embeddings.create(
        input=question, model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def create_faq_embedding():
    qa_pairs = json.loads(open("FAQ.json").read())[:41]
    # [
    #    {
    #        "question": "...",
    #        "answer": "..."
    #    }
    # ]
    df = pd.DataFrame(qa_pairs)
    df['embedding'] = df['question'].apply(get_embedding)
    df.to_json('embeddings.json', orient='records')

def load_faq():
    df = pd.read_json('embeddings.json')
    return df

def search_faq(df, user_question):
   user_question_embedding = get_embedding(user_question)

   # create new pandas frame without the embedding column and add a new column for similarities
   df['similarities'] = df.embedding.apply(lambda x: cosine_similarity(x, user_question_embedding))
   res = df.sort_values('similarities', ascending=False)
   res = res.drop('embedding', axis=1)
   print(colored(res, 'grey'))
   return res

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    df: pd.DataFrame,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    related_faqs: pd.DataFrame = search_faq(df, query)
    introduction = 'Use the below official FAQs to answer the subsequent question. If the answer cannot be found in the articles, write „I could not find an answer.“'
    question = f"\n\nQuestion: {query}"
    message = introduction
    # iterate faqs beginning with the most similar, max 5
    max_faqs = 5
    for faq in related_faqs.itertuples():
        next_article = f'\n\nOfficial FAQ section:\n"""\nQuestion: {faq.question}\n\n\nAnswer: {faq.answer}\n\n"""'
        if (
            num_tokens(message + next_article + question, model=GPT_MODEL)
            > token_budget
        ):
            break
        else:
            message += next_article

        max_faqs -= 1
        if max_faqs == 0:
            break

        # break # only one article for now
    return message + question

def ask(
    query: str,
    df: pd.DataFrame,
    token_budget: int = 4096 - 500,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, token_budget=token_budget)
    print(colored(f"\n\n---\n{message}\n---\n\n", 'grey'))
    messages = [
        {"role": "system", "content": "You answer questions for students from the TUM School of Management."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content # response["choices"][0]["message"]["content"]
    return response_message


# print(search_reviews(df, "Is part-time study possible?"))

# create_faq_embedding()
# exit()

df = load_faq()

while True:
    user_question = input(colored("Ask a question: ", 'green'))
    res = ask(user_question, df)
    if res == "I could not find an answer.": 
        print(colored(res, 'red'))
    else:
        print(colored(res, 'green'))
    print("\n---\n\n")






