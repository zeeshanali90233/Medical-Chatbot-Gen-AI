from flask import Flask,render_template,jsonify,request
from dotenv import load_dotenv
from src.helper import load_llm,get_pinecone_retreiver,make_qa_chain,generate_response,download_embed_model
import logging
logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')
load_dotenv()


app=Flask(__name__)


model_path="model\\llama-2-7b-chat.ggmlv3.q4_0.bin"
llm=load_llm(model_path)
logging.info(f"LLM Loaded")
embedding=download_embed_model()
logging.info(f"Embedding Loaded")
retriever=get_pinecone_retreiver("medical-chatbot",embedding_model=embedding)
logging.info(f"Pinecone Retriever Loaded")
qa_chain=make_qa_chain(llm=llm,retriever=retriever)
logging.info(f"Chain Made")
logging.info(f"Generating Response")

# print(generate_response(qa_chain,"Name the books name from you are taking help while responding?"))


@app.route("/get",methods=['GET',"POST"])
def chat():
    chat = request.form['msg']
    response = generate_response(qa_chain, chat)
    return render_template("chat.html", response=response)


@app.route("/")
def index():
    return render_template("chat.html")



if __name__=="__main__":
    app.run(debug=True)