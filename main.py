import openai
from openai import OpenAI
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
import psycopg2
from utils import Config, get_openai_api_key
import json
import sys
import os



# Set OPENAI_API_KEY environment variable
os.environ['OPENAI_API_KEY'] = get_openai_api_key()




def json_query_embedding(query, config):
    """
    Fetch the vector embedding for a given query from a JSON file.
    """
    try:
        with open(config.json_path, 'r') as file:
            questions = json.load(file)
        return questions.get(query)
    except FileNotFoundError:
        print(f"The file {config.json_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {config.json_path}.")
        return None



def postgreSQL_search(embedding, config):
    """
    Retrieve context from PostgreSQL based on the closest embedding vector based on vector distance.
    """
    try:
        if config.dist_metric == 'L2':
            dist_metric = '<->'
        elif config.dist_metric == 'cosine':
            dist_metric = '<=>'
        else:
            print("No valid distance metric chosen!")
            return []
        
        connect_str = f"dbname={config.dbname} user={config.user}"
        with psycopg2.connect(connect_str) as conn:
            with conn.cursor() as cur:
                sql = """
                SELECT text FROM embeddings
                ORDER BY embedding_vector {0} '{1}'
                LIMIT {2};
                """.format(dist_metric, embedding, config.n_context)
                cur.execute(sql) 
                results = cur.fetchall()
                return results
    except Exception as e:
        print(f"Database operation failed: {e}")
        return []
    



def llm_rag(query, context, config):
    """
    Generate a response based on the query and the context using a language model.
    """
    try:
        template = """
        You are an assistant to answer questions based on a given context. 
        If the question cannot be answered exactly based on the context or is ambiguous, respond with "I cannot answer based on the context". 
        Do not hallucinate!

        Question:
        {query}

        Context: 
        {context}

        Answer:
        """
        prompt = PromptTemplate(template=template, input_variables=["query", "context"])

        if config.model == 'gpt-4':
            model = ChatOpenAI(model=config.model, temperature=config.temperature)
        elif config.model == 'gpt-3.5-turbo-instruct':
            model = OpenAI(model=config.model, temperature=config.temperature)
        else:
            print('Chosen model unknown!')
            return []
        
        chain = prompt | model | StrOutputParser()
        response = chain.invoke({"query": query, "context": context})
        return response
    except Exception as e:
        print("Error during LLM RAG processing:", e)
        return None



def rag_pipeline(query):
    """Executes the Retrieval-Augmented Generation (RAG) pipeline."""
    config = Config('config.yml')
    embedding = json_query_embedding(query, config)
    context = postgreSQL_search(embedding, config)
    response = llm_rag(query, context, config)
    return context, response


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python main.py <question> <flag context>")
        sys.exit(1)

    _, response = rag_pipeline(args[0])
    print(response)


if __name__ == "__main__":
    main()

