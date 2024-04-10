import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pymongo
from htmlTemplates import css, bot_template, user_template
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
from dotenv import load_dotenv
load_dotenv()
device = 'cuda'
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def embed(text):
    emb_model = SentenceTransformer("thenlper/gte-large")
    if not text.strip():
        print("Attempted to embed an empty string")
        return []
    embedding = emb_model.encode(text)

    return embedding.tolist()

def embed_chunks(chunks):

    texts_n_embs = []
    for text in chunks:
        item = {}
        item['text'] = text
        item['embedding'] = embed(text)
        texts_n_embs.append(item)

    return texts_n_embs

def get_mongo_client(mongo_uri):
    try:
        client = pymongo.MongoClient(mongo_uri)
        print("[SUCCESS] Connected to MongoDB")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"[FAILED] Connection failed: {e}")
        return None
    

mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    print("MONGO_URI not in env")

mongo_client = get_mongo_client(mongo_uri)

db = mongo_client["my_db"]
collection = db["my_collection"]

def add_embs_to_db(texts_n_embs):    
    collection.delete_many({})
    collection.insert_many(texts_n_embs)


def vector_search(query, collection):

    query_embedding = embed(query)

    # define the vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,  # Number of candidate matches to consider
                "limit": 5,  # Return top n matches
            }
        },
        {
            "$project": {
                "_id": 0,  # 0 -> exclude
                "text": 1, # 1 -> include
                "score": {"$meta": "vectorSearchScore"},  # include the search score
            }
        },
    ]

    results = collection.aggregate(pipeline)

    return list(results)

def get_context(query, collection):

    results = vector_search(query, collection)
    context_items = ""
    for result in results:
        context_items += f"{result.get('text', 'N/A')}\n"

    return context_items

def handle_userinput(query):
    context = get_context(query, collection)
    base_prompt = """You are a helpful assisstant.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as clear and concise.
    \nNow based on the following context items:
    {context};
    \n And answer the user's query:
    User query: <start_of_turn>user{query}<end_of_turn>
    Model: answer:"""
    base_prompt = base_prompt.format(context=context, query=query)
    dialogue_template = [
        {
            'role': 'user',
            'content': base_prompt
        }
    ]
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    response = model.generate(**input_ids, max_new_tokens=500, temperature=0.7, do_sample=True)
    text = tokenizer.decode(response[0])
    st.write(user_template.replace(
                "{{MSG}}", text), unsafe_allow_html=True)
    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    #     else:
    #         st.write(bot_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)

model_id = 'google/gemma-2b'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)


def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    query = st.text_input("Ask a question about your documents:")
    if query:
        handle_userinput(query)

    with st.sidebar:
        st.subheader("Your documents")
        # pdf_docs = st.file_uploader(
            # "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        # if st.button("Process"):
        #     with st.spinner("Processing"):
        #         raw_text = get_pdf_text(pdf_docs)
        #         chunks = get_text_chunks(raw_text)
        #         texts_n_embs = embed_chunks(chunks)
        #         add_embs_to_db(texts_n_embs)



if __name__ == '__main__':
    main()
