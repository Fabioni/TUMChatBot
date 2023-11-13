Put your OpenAI API Key in keys.py

Use extractFAQ.js to scrap all FAQs from the website

Use `create_faq_embedding()` from main.py to get the embeddings for it. Only use the first 41 FAQs since they all belong to the same Study Program. Different Programs are not supported yet. They would have to be included in the embedding creation process

run main.py to ask questions and get answers.