from dotenv import load_dotenv
from src.helper import load_llm,get_pinecone_retreiver,make_qa_chain,generate_response,download_embed_model
import logging
logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')
load_dotenv()


model_path="E:\\transformers_cache\\models--TheBloke--Llama-2-7B-Chat-GGML\\snapshots\\76cd63c351ae389e1d4b91cab2cf470aab11864b\\llama-2-7b-chat.ggmlv3.q4_0.bin"
llm=load_llm(model_path)
logging.info(f"LLM Loaded")
embedding=download_embed_model()
logging.info(f"Embedding Loaded")
retriever=get_pinecone_retreiver("medical-chatbot",embedding_model=embedding)
logging.info(f"Pinecone Retriever Loaded")
qa_chain=make_qa_chain(llm=llm,retriever=retriever)
logging.info(f"Chain Made")
logging.info(f"Generating Response")

print(generate_response(qa_chain,"Which Drug is Effective against anaerobic protozoan parasites?"))
