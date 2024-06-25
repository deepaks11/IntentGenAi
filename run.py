import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


c_w_d = os.getcwd()
dataset = os.path.join(c_w_d, "dataset/current_cp_dataset.csv")
model_path = os.path.join(c_w_d, "models/openchat-3.5-0106-AWQ")

loader = CSVLoader(file_path=dataset, encoding="utf-8", csv_args={'delimiter': ','})
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=64,
    separators=['\n\n', '\n', '(?=>\. )', ' ', '']
)

docs = text_splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/average_word_embeddings_glove.6B.300d")

llm = HuggingFacePipeline.from_model_id(
    model_id=model_path,
    task="text-generation",
    device=0,
    # device_map="auto",  # replace with device_map="auto" to use the accelerate library.
    pipeline_kwargs={"max_new_tokens": 20},
    model_kwargs={"low_cpu_mem_usage": True}
)


text_llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3", task="text-generation",
                          model_kwargs={"temperature": 1, "max_length": 20})

text_chain = load_qa_chain(text_llm, chain_type="stuff")

chain = load_qa_chain(llm, chain_type="stuff")

db = FAISS.from_documents(docs, embeddings)

actions = ["run", "start", "stop", "write", "delete", "create", "edit", "move", "activate", "deactivate"]

while True:

    question = input("Enter your question: ")
    s = time.time()
    docs_ = db.similarity_search(question)

    result = ["yes" if action in question else "no" for action in actions]

    if "yes" in result:

        prompt = f"""When suggesting intents for the user, ensure to follow these guidelines 

                Here's an example dialogue step by step you must follow below format
                1. Answer: 'At present, I'm unable to execute the requested task, but I can provide information on related topics.'
                2. Related Topics: Provide the intents only based on the following documents

                Please structure your response accordingly."""
    else:

        text = text_chain.run(
            input_documents=docs_,
            question=f"""Please note carefully jus give the detail and information about the user question, If the question aligns with any intents, respond with yes and shortly answer
                     the question appropriately, for example you can get and give the details, list , report. Here's the question: {question}, don't generate to much"""
        )

        text_response = text.split("Helpful Answer:")[1].split("\n\n")[0].strip()

        try:
            ans = chain.run(
                input_documents=docs_,
                question=question + ", give me the intent only"
            )

            intent = ans.split("Helpful Answer:")[1]

        except:

            intent = ans

        prompt = f"""When suggesting intents for the user, ensure to follow these guidelines 

                Here's an example dialogue step by step you must follow below format

                1.Related Topics: Provide the intents only based on the following documents

                Please structure your response accordingly."""

    try:

        ans = chain.run(
            input_documents=docs_,
            question=prompt
        )

        suggestion = ans.split("Helpful Answer:")[1].split("1.")[1]

    except:

        suggestion = ans

    if "yes" in result:

        [print(out) for out in ans.split("Helpful Answer:")[1].split(".")[1:4:2]]

    else:
        final_response = f"Response: {text_response} \n\nIdentified Intent: {intent} \n\nSuggestion: {suggestion}"
        print(final_response)
        # [print(re.sub("[0-9]", "", out)) for out in ans.split("Helpful Answer:")[1].split(".")[2:]]

        print("execute time :", time.time() - s)
