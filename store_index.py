from src.helper import load_pdf, load_llm,split_texts,download_embed_model,store_chunks_embeddings
import logging
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')

data=load_pdf("./data")
logging.info(f"Extracted Data")
text_chunks=split_texts(data)
logging.info(f"Chunks Made")
embedding=download_embed_model()
logging.info(f"Embedding Model Downloaded")

retriever=store_chunks_embeddings(text_chunks,"medical-chatbot",384,embedding_model=embedding)
logging.info(f"Saved the embeddings")
