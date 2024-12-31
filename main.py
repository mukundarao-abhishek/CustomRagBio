import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from llama_cpp import Llama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import re

# Define classes and functions

def load_pdf(file_path):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

MARKDOWN_SEPARATORS = [
    r"\n#{1,6} ",
    r"```\n",
    r"\n\*\*\*+\n",
    r"\n---+\n",
    r"\n___+\n",
    r"\n\n",
    r"\n",
    r" ",
    r"",
]

def split_with_overlap(text, max_length, overlap):
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_length
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

def split_text_recursive(text, separators=MARKDOWN_SEPARATORS, max_length = 512, overlap = 51):
    
    if not separators:
        return split_with_overlap(text, max_length, overlap)

    separator = separators[0]
    parts = re.split(separator, text)
    chunks = []
    current_chunk = ""

    for part in parts:
        if len(current_chunk) + len(part) + len(separator) <= max_length:
            current_chunk += (separator + part) if current_chunk else part
        else:
            if current_chunk:
                if len(current_chunk) > max_length:
                    chunks.extend(split_with_overlap(current_chunk, max_length, overlap))
                else:
                    chunks.append(current_chunk)
            if len(part) + len(separator) > max_length:
                smaller_chunks = split_text_recursive(part, separators[1:], max_length, overlap)
                chunks.extend(smaller_chunks)
            else:
                current_chunk = part

    if current_chunk:
        if len(current_chunk) > max_length:
            chunks.extend(split_with_overlap(current_chunk, max_length, overlap))
        else:
            chunks.append(current_chunk)

    return chunks

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

class Embedder:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def generate_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            model_output = self.model(**inputs)
            sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

def retrieve_documents(query_embedding, document_embeddings, documents, top_k=5):
    similarities = cosine_similarity(query_embedding, document_embeddings)
    top_k_indices = similarities.argsort()[0][-top_k:][::-1]
    return [documents[i] for i in top_k_indices]


def format_prompt(context, question):
    prompt_in_chat_format = [
        {
            "role": "user",
            "content": """Using the information contained in the context,
                        give a comprehensive answer to the question.
                        Respond only to the question asked, response should be concise and relevant to the question.
                        Provide the number of the source document when relevant.
                        If the answer cannot be deduced from the context, do not give an answer.""",
        },
        {
          "role": "assistant",
          "content" : "Sure I'll follow it."  
        },
        {
            "role": "user",
            "content": f"""Context:{context}
                        ---
                        Now here is the question you need to answer.
                        Question: {question}""",
        },
    ]
    return prompt_in_chat_format

class Generator:

    def __init__(self, model_name, model_file):
        self.model = Llama.from_pretrained(
                    repo_id=model_name,
                    filename=model_file,
                    n_ctx=2048,
                    verbose=False,
                )
    
    def generate(self, query, context):
        final_prompt = format_prompt(context=context,question=query)
        answer = self.model.create_chat_completion(final_prompt)
        return answer['choices'][0]['message']['content']
    
# from huggingface_hub import login
# login()

# Streamlit app

st.title("Custom RAG Pipeline with Mistral-7B-Instruct-v0.1")

st.write("Loding the model...")

generator = Generator('TheBloke/Mistral-7B-Instruct-v0.1-GGUF','mistral-7b-instruct-v0.1.Q4_K_M.gguf')

st.write("Upload a PDF, ask a question, and get a generated response based on the document content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_path = uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Processing the PDF...")
    pdf_text = load_pdf(pdf_path)
    chunks = split_text_recursive(pdf_text)
    
    st.write("Generating embeddings for the document...")
    embedder = Embedder('thenlper/gte-small')
    document_embeddings = embedder.generate_embeddings(chunks)
    
    st.write("You can now ask a question about the document.")
    query = st.text_input("Enter your question:")
    
    if query:
        st.write("Generating response...")
        query_embedding = embedder.generate_embeddings([query])
        relevant_docs = retrieve_documents(query_embedding, document_embeddings, chunks)
        
         # Create Context string
        context = ""
        context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])
        
        response = generator.generate(query=query,context=context)
        
        st.write("Response:")
        st.write(response)
