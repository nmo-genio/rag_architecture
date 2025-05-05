import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_ATLAS_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME")
VECTOR_INDEX = os.getenv("MONGODB_SEARCH_INDEX")

assert OPENAI_API_KEY, "OPENAI_API_KEY not set in environment"
assert MONGODB_URI, "MONGODB_ATLAS_URI not set in environment"
assert DB_NAME, "MONGODB_DB_NAME not set in environment"
assert COLLECTION_NAME, "MONGODB_COLLECTION_NAME not set in environment"
assert VECTOR_INDEX, "MONGODB_SEARCH_INDEX not set in environment"

# Test MongoDB connection. If successful, show availa
try:
    test_client = MongoClient(
        MONGODB_URI,
        tls=True,
        tlsAllowInvalidCertificates=True,
    )
    # test_db_names = test_client.list_database_names()
    # print("✅ Connected to MongoDB. Available databases:", test_db_names)
    print("✅ Connected to MongoDB")
except Exception as e:
    print("❌ Failed to connect to MongoDB:", e)

vectorStore = MongoDBAtlasVectorSearch.from_connection_string(
    MONGODB_URI,
    DB_NAME + "." + COLLECTION_NAME,
    OpenAIEmbeddings(disallowed_special=(), openai_api_key=OPENAI_API_KEY),
    index_name=VECTOR_INDEX,
)


def query_data(query: str):
    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3
        },
    )

    template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Do not answer the question if there is no given context.
        Do not answer the question if it is not related to the context.
        Do not give recommendations to anything other than MongoDB.
        Context:
        {context}
        Question: {question}
        """

    custom_rag_prompt = PromptTemplate.from_template(template)

    # this is a dictionary
    retrieve = {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    }

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    response_parser = StrOutputParser()

    rag_chain = (
        retrieve
        | custom_rag_prompt
        | llm
        | response_parser
    )
# Logic when working without UI
    # answer = rag_chain.invoke(query)
    # return answer

# print(query_data("What is the difference between a collection and database in MongoDB?"))
# print(query_data("Why is the sky blue"))

# Here is where we integrate with UI
# if __name__ == "__main__":
#     while True:
#         user_query = input("Ask me a question about MongoDB (or type 'exit' to quit): ")
#         if user_query.lower() in ["exit", "quit"]:
#             print("Goodbye! ")
#             break
#         print(query_data(user_query))


# Logic when working with UI
    return rag_chain.invoke(query)
