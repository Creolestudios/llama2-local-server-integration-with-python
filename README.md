# llama2-local-server-integration-with-python

Welcome to the LLaMA-2 Medical Bot Project! This project demonstrates the installation and utilization of Meta's LLaMA-2, a cutting-edge large language model (LLM), to create an interactive medical chatbot. The primary objective of this project is to guide you through the process of setting up the LLaMA-2 model and integrating it into a simple demo that can provide basic medical information and support.

## Table of Contents

- [Introduction](#llama2-local-server-integration-with-python)
- [Table of Contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Setup](#setup)
- [Usage](#usage)

## Prerequisites

Before you can start using the Llama2 Medical Bot, make sure you have the following prerequisites installed on your system:

- Python 3.6 or higher
- Required Python packages (you can install them using pip):
  - langchain
  - chainlit
  - sentence-transformers
  - faiss
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

With these steps, you can set up and use the LLaMA-2 model to create an interactive medical chatbot. For more detailed information, refer to the documentation and examples provided within the project.
