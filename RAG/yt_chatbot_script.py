##Youtube chatbot, allowing to chat with yt videos.

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1 : Indexing(Document Ingestion)
video = input("Enter the video ID: ")

video_id = video  
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id,languages=['hi'])##gives transcript with timestapms and other data
    transcript = " ".join(chunk['text'] for chunk in transcript_list) ##flattening it to a plain text,i.e only the subtitles.
    print(transcript[:100])
except TranscriptsDisabled:
    print("No transcript available for the video you provided.")

# STEP2:SPLITTING THE TEXT INTO CHUNKS

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = splitter.create_documents([transcript])

len(chunks)

# STEP:3 ENBEDDING GENERATION AND STORING IN VECTOR STORE

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  
)

vector_store = FAISS.from_documents(chunks,embedding_model)

vector_store.index_to_docstore_id

# STEP 4: Retrieval

retriever = vector_store.as_retriever(search_types = "similarity", search_kwargs={"k":4})

# STEP 5 AUGMENTATION

prompt = PromptTemplate(
    template = """You are a Helpful assistant. Answer only from the provided transcript context.
    If the context is insufficient, just say you do not know.
    {context}
    Question:{question}""",
    input_variables =  ['context','question']    
)
llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

from langchain_core.runnables import RunnableParallel,RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(func = format_docs),
    'question': RunnablePassthrough()
}
)

parser = StrOutputParser()

main_chian = parallel_chain | prompt | llm | parser
while True:
    user_input = input("Enter Your query: ")
    if user_input in ['Exit','exit','bye','quit']:
        break
    response = main_chian.invoke(user_input)
    print(response)


