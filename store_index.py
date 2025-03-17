from src.helper import load_pdf_files, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

extracted_data = load_pdf_files(data='./data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-genai-bot"
existing_indexes = [index_info.name for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=CloudProvider.AWS,
            region=AwsRegion.US_EAST_1
        )
    )

docsearch = PineconeVectorStore.from_documents(documents=text_chunks,index_name=index_name,embedding=embeddings)
