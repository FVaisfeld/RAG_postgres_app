# RAG Application for Predefined Question and Context Embeddings


## Introduction

   This project implements a Retrieval-Augmented Generation (RAG) chain designed to process a given question, retrieve relevant 
   context from a PostgreSQL database, and generate an answer based on the context. The core components of this project include:

      - Extracting pre-calculated embeddings from a JSON file based on the input question.
      - Retrieving context by comparing the distance between pre-calculated context embeddings and the question embedding.
      - Generating a response using a language model based on the question and retrieved context information.



## Prerequisites

Before running the application, ensure the following prerequisites are met:

   - A PostgreSQL database containing the context embeddings must be set up.
   - Pgvector needs to be installed to enable vector-distance measures.



## Setup and Installation

   ### Dependencies
      Install the required dependencies specified in `requirements.txt` e.g. via command line by executing:

         pip install -r requirements.txt

   ### Configuration
      1. Obtain an API key from the OpenAI website for communicating with the LLMs. Place this key in an .env file within the application 
         directory as follows:

         openaiAPI=<Your_OpenAI_API_Key>

      2. Configure the PostgreSQL database details (name and user) in the config.yml file. Other parameters are available for customization, but   the       default values should suffice for a functioning pipeline.

   ### Execution
      Run the application via the command line by executing:

         python main.py "What is the revenue of Alphabet Inc?"
      
      Ensure you enter one of the four questions available in the JSON file, as embeddings are pre-calculated only for these questions.



## Evaluation

   The application can be evaluated using the provided eval_notebook, which leverages the RAGAS toolbox for evaluation. Evaluation questions should be specified in the eval_questions.txt file. Currently, the application supports only the four questions for which embeddings are available in the JSON file. The metrics calculated with RAGAS for the RAG application include:

      - Faithfulness: How factually accurate is the answer based on the context.
      - Answer relevancy: How relevant is the generated answer to the question
      - Context Relevancy: How relevant is the context to the question

   ### Observations: 
   - Switching from GPT-3-turbo-instruct to GPT-4 resulted in a significant increase in faithfulness.
   - The context relevance is relatively low. Consider implementing a preprocessing step that ranks/filters the context to reduce noise sent to the   LLM  or fine-tuning an embedding model.
   - No significant difference was observed when changing the retrieved context embedding distance measure from 'L2' to 'cosine'.# RAG_postgres_app
