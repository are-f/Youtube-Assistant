from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Globals to hold the chain
main_chain = None

@app.route("/init", methods=["POST"])
def init():
    global main_chain

    data = request.json
    video_id = data.get("video_id")

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi"])
        transcript = " ".join(chunk['text'] for chunk in transcript_list)
    except TranscriptsDisabled:
        return jsonify({"error": "Transcript is disabled or unavailable."}), 400

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template="""You are a Helpful assistant. Answer only from the provided transcript context.
If the context is insufficient, just say you do not know.
{context}
Question: {question}""",
        input_variables=['context', 'question']
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(func=format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    return jsonify({"message": "Transcript loaded and chatbot ready!"})


@app.route("/chat", methods=["POST"])
def chat():
    global main_chain
    if main_chain is None:
        return jsonify({"error": "Chatbot not initialized."}), 400

    user_query = request.json.get("query")
    response = main_chain.invoke(user_query)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
