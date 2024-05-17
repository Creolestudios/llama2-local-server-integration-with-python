# Llama-2 Local Server Integration with Python

Welcome to the Llama-2 Medical Bot Project! This project demonstrates the installation and utilization of Meta's LLaMA-2, a cutting-edge large language model (LLM), to create an interactive medical chatbot. The primary objective of this project is to guide you through the process of setting up the LLaMA-2 model and integrating it into a simple demo that can provide basic medical information and support.

## Table of Contents

- [Introduction](#llama2-local-server-integration-with-python)
- [Table of Contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Setup](#setup)
- [Usage](#usage)
- [Prompts](#prompts)
- [Vector Database](#vector-database)
- [Further Development](#further-development)

## Prerequisites

Before you can start using the Llama2 Medical Bot, make sure you have the following prerequisites installed on your system:

- Python 3.6 or higher
- Required Python packages (you can install them using pip):
  - [Langchain](https://www.langchain.com/): A library for natural language processing tasks such as document loading, embeddings, and vector stores.
  - [Chainlit](https://github.com/langchain/chainlit): A framework for building conversational agents and chatbots.
  - [HuggingFace Transformers](https://github.com/huggingface/transformers): A library that provides state-of-the-art natural language processing models and embeddings.
  - [FAISS](https://github.com/facebookresearch/faiss): A library for efficient similarity search and clustering of dense vectors.
  - PyPDF2 (for PDF document loading)
- Llama-2 model installed in your local machine

## Installation

Here's how to get the LLaMA-2 (TheBloke/Llama-2-7B-Chat-GGML) model.

First you need to go to [HuggingFace](#https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML) and fill out the form to request access.

After your request is approved, you will receive an email. Then, go to the [files section](#https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main) and download the file [llama-2-7b-chat.ggmlv3.q8_0.bin](#https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin)

Once you have downloaded this file, you are ready to use the LLaMA-2 model on your local machine.

## Getting Started

To get started with the LLaMA-2 Medical Bot:

1. Set up your environment and install the required packages as described in the prerequisites.

2. Create a folder in your project to store the downloaded model file.

3. Prepare the language model and data as per the Langchain documentation.

## Setup

```
#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm
```

## Usage

Here's how you can use the model to query medical information:

```
#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa
```

Retrieval chain from langchain:

```
#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain
```

## Prompts

```
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt
```

## Vector Database

Here's how you can create vector database:

```
#create vector database

def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob = "*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device' : 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
```

## Further Development

To further enhance this project, consider the following suggestions:

1. Improving the User Experience: Enhance the chatbot interface with features like sentiment analysis and natural language understanding.
2. Expanding the Knowledge Base: Incorporate additional medical datasets to improve the chatbot's knowledge and accuracy.
3. Integrating with External Systems: Integrate the chatbot with external systems such as Electronic Health Records for personalized responses.
4. Implementing Continuous Learning: Implement mechanisms for the chatbot to learn from user interactions and feedback.
5. Scaling for Production: Optimize the deployment architecture for scalability and reliability in production environments.

With these steps, you can set up and use the LLaMA-2 model to create an interactive medical chatbot. For more detailed information, refer to the code provided within the project.
