from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
from pinecone import Pinecone,ServerlessSpec
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate


#Extract Data
def load_pdf(data_dir):
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs


def split_texts(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(docs)
    return texts

# Download Embedding Model
def download_embed_model():
    embeddings=HuggingFaceEmbeddings(model_name=os.environ.get("EMBEDDING_MODEL_NAME"))
    return embeddings

def store_chunks_embeddings(chunks,index_name,index_dimension,embedding_model):
    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
    indexExists=False
    for index_detail in pc.list_indexes():
        if(index_detail.name==index_name):
            indexExists=True

    if(not indexExists):
        pc.create_index(
        name=index_name,
        dimension=index_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
        )
    
    vector_store=PineconeVectorStore(index=index_name,embedding=embedding_model)
    vector_store.from_texts([text.page_content for text in chunks],embedding=embedding_model,index_name=index_name)
    return vector_store.as_retriever(search_kwargs={"k":5})

def get_pinecone_retreiver(index_name,embedding_model):
    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
    index = pc.Index(index_name)
    vector_store=PineconeVectorStore(index=index,embedding=embedding_model)
    return vector_store.as_retriever(search_kwargs={"k":2})


def load_llm(model_path):
    llm=CTransformers(model=model_path, model_type="llama", config={
    "temperature": 0.8,      
    "max_new_tokens":512
    })
    return llm

def make_qa_chain(llm, retriever):
    chain = RetrievalQA.from_chain_type(
        llm=llm,  
        retriever=retriever,
        chain_type="stuff", 
        chain_type_kwargs={"prompt":make_prompt_template()}, 
        return_source_documents=True
    )
    return chain

def generate_response(chain,query):
    res=chain.invoke(query)
    # print(res['source_documents'])
    return res['result']

def make_prompt_template():
    medical_template = PromptTemplate(
        template="""
        You are a very knowledgable , helpful honest medical assistant . Use the following context to answer the user's medical-related question.

        - Only provide an answer if the question is medical in nature.
        - If the question is not related to medical topics, kindly refuse to answer.
        - If you do not know the answer, it is okay to say, "I don't know."

        Context: {context}
        Question: {question}
        
        Provide a clear, concise, and accurate medical response below:
        Answer:
        """,
        input_variables=['context', 'question']
    )

    return medical_template
