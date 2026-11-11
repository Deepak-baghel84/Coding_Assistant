import os
import sys
from pathlib import Path
from typing import List
import json
from dotenv import load_dotenv
from groq_eval_model import GroqEvalModel
import time
import pandas as pd

#from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
)
from deepeval import evaluate

from logger import GLOBAL_LOGGER as log

from src.document_ingestion.data_ingestion import ChatIngestor
from src.document_chat.retrieval import ConversationalRAG
from collections import defaultdict
from groq_eval_model import GroqEvalModel


eval_model = GroqEvalModel(
    model_name="openai/gpt-oss-120b"
)


        # Directory containing documents to ingest and index(raw data for evaluation)
DEEPEVAL_INPUT_DIR = os.getenv("DEEPEVAL_INPUT_DIR","data_deep_eval")

             # Document versioning path
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "Data/multidoc_archive")    

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")   # faiss index path
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")

       # Local JSON evaluation dataset
DEEPEVAL_DATASET_PATH = os.getenv("DEEPEVAL_DATASET_PATH", "eval_ques/omnibench_rag_evaluation_dataset_formatted.json")  # dataset path, questions and expected answers for evaluation





Latency = {"Ingestion":[],"Retrieval":[],"Generation":[],"Evaluation":[]}

Tokens = {"input_token":[],"output_token":[],"cache_token":[]}





class LocalFileAdapter:
    def __init__(self, file_path: str):
        self.name = os.path.basename(file_path)
        self._file_path = file_path

    def read(self) -> bytes:
        with open(self._file_path, "rb") as f:
            return f.read()

    # For compatibility with save_uploaded_files
    def getbuffer(self) -> bytes:
        return self.read()


def list_supported_files(root: Path) -> List[Path]:
    exts = {".pdf", ".docx", ".txt", ".pptx", ".md", ".csv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3"}
    files: List[Path] = []
    for p in sorted(root.rglob("*")):    # Take the root directory and recursively search for all files, then sort them alphabetically.
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files




def query_rag(question: str,rag: ConversationalRAG) -> dict:

    result = rag.invoke_with_context(
        question=question,
        chat_history=[],
        Latency=Latency
    )


    documents = result["documents"]

    context = [doc.page_content for doc in documents]

    print(f"RETRIEVED {len(documents)} DOCUMENTS")

    answer= result["answer"]
    tokens = getattr(answer, "usage_metadata", None)
    Tokens["input_tokens"].append(tokens.get("input_tokens"))
    Tokens["output_tokens"].append(tokens.get("output_tokens"))
   

   # _log_docs(documents)  # Log the retrieved documents

    return {
        "answer": result["answer"],
        "context": context
    }



def load_local_dataset(dataset_path: str) -> list[dict]:   # seprates test cases and metadata and returns only the test cases
    """
    Load evaluation questions and expected answers
    from a local JSON dataset.
    """

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Supports the recommended structure:
    #
    # {
    #     "metadata": {...},
    #     "test_cases": [...]
    # }

    test_cases = data.get("test_cases", [])

    if not test_cases:
        raise ValueError(
            "No test cases found in evaluation dataset."
        )

    return test_cases

def run_metric_with_retry(test_case, metric, max_retries=3):

    for attempt in range(max_retries):

        try:

            result = evaluate(
                test_case=test_case,
                metric=metric
            )

            print("type",type(result))
            print("result",result)
            return result

        except Exception as e:

            error_message = str(e)

            print(
                f"Attempt {attempt + 1}/{max_retries} failed: "
                f"{error_message}"
            )

            if attempt == max_retries - 1:
                raise

            wait_time = 30 * (attempt + 1)

            print(
                f"Waiting {wait_time} seconds before retry..."
            )

            time.sleep(wait_time)


def main():
    # Load env locally like in the notebook 
    if os.getenv("ENV", "local").lower() != "production":
        load_dotenv()
        log.info("Running in LOCAL mode: .env loaded")

    # 1) Build or load FAISS index from the specified directory
    data_dir = Path(DEEPEVAL_INPUT_DIR)          
    assert data_dir.exists(), f"Input dir not found: {data_dir}"
    paths = list_supported_files(data_dir)  # List all supported files in the input directory(raw data)
    if not paths:
        log.error("No supported files found in input directory", dir=str(data_dir))
        print("No supported files found in input directory.")
        sys.exit(1)

    # Ingest and index
    start = time.perf_counter()

    chat_ingestor = ChatIngestor(temp_base=UPLOAD_BASE, faiss_base=FAISS_BASE,use_session_dirs = True,session_id= None) #raw dataset
    adapters = [LocalFileAdapter(str(p)) for p in paths]
    retriever =  chat_ingestor.create_retrivel(adapters,chunk_size= 500,chunk_overlap= 100,k= 2)
    latency = time.perf_counter() - start
    Latency["Ingestion"].append(latency)
    log.info("Ingestion complete", session_id=chat_ingestor.session_id)

   

    rag = ConversationalRAG(
    session_id=chat_ingestor.session_id
)

 #create retrievel only once and use it for all queries

    index_dir = os.path.join(FAISS_BASE, chat_ingestor.session_id)
    rag.load_retriever_from_faiss(index_path=index_dir,k=2,index_name=FAISS_INDEX_NAME) 

      

    # 2) Pull dataset from Confident AI (cloud service for deepeval)
    
    # dataset = EvaluationDataset()
    # dataset.pull(alias=DATASET_ALIAS)

    # -------------------------------------------------
    # 3) Load local evaluation dataset(goldens)
    # -------------------------------------------------

    local_dataset = load_local_dataset(
        DEEPEVAL_DATASET_PATH
    )

    log.info(
        "Local evaluation dataset loaded",
        total_test_cases=len(local_dataset)
    )

    deepeval_test_cases = []

    stop = 0
    for item in local_dataset:

        if stop >0:  # Limit to 1 test cases for demonstration purposes
            break
        stop += 1

        question = item.get("input")
        expected_output = item.get("expected_output")
        expected_context = item.get("expected_context", [])

        if not question or not expected_output:

            log.error(
                "Invalid dataset item",
                item=item
            )

            continue

        try:

            result = query_rag(
                question,
                rag=rag
            )
            log.info("Query executed successfully",
                question=question,
                answer=result["answer"],
            )
            # Ensure context is a list
            retrieved_context = result["context"]

            if isinstance(retrieved_context, str):

                retrieved_context = [
                    retrieved_context
                ]

            # log.info("Retrieved context and test case creation started")

            test_case = LLMTestCase(
                input=question,

                actual_output=result["answer"],

                expected_output=expected_output,

                retrieval_context=retrieved_context,

                context=expected_context,             # we considering both retrieval context and context as the same for evaluation purpose which effect the evaluation metrics response.
            )

            deepeval_test_cases.append(
                test_case
            )

            log.info(
                "Test case created successfully",
                question=question
            )

        except Exception as e:

            log.error(
                "Failed to build test case",
                error=str(e),
                question=question,
            )


    time.sleep(10)  # wait for a minute to avoid rate limiting issues with the evaluation model

    # -------------------------------------------------
    # 5) Check test cases
    # -------------------------------------------------

    if not deepeval_test_cases:

        raise RuntimeError(
            "No DeepEval test cases were created."
        )
    


    

    from deepeval import evaluate
    from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig


    metrics = [
    AnswerRelevancyMetric(
        model=eval_model,
        async_mode=False
    ),

    FaithfulnessMetric(
        model=eval_model,
        async_mode=False
    ),

    ContextualPrecisionMetric(
        model=eval_model,
        async_mode=False
    ),

    ContextualRecallMetric(
        model=eval_model,
        async_mode=False
    )
    ]


    print(
    f"Starting evaluation: "
    f"{len(deepeval_test_cases)} goldens × "
    f"{len(metrics)} metrics"
    )


    evaluation_results = evaluate(
    test_cases=deepeval_test_cases,
    metrics=metrics,

    async_config=AsyncConfig(
        run_async=False
    ),

    cache_config=CacheConfig(
        write_cache=False,
        use_cache=False
    ),

    display_config=DisplayConfig(
        show_indicator=True,
        print_results=False,
        inspect_after_run=False
    )
)


    print("Evaluation completed!")
    
    for test_result in evaluation_results.test_results:
        print(f"\nTest Case: {test_result.name}")
        for metric_data in test_result.metrics_data:
            print(
            f"{metric_data.name}: "
            f"{metric_data.score:.2f} "
            f"({'PASS' if metric_data.success else 'FAIL'})"
            )

    print("="*60)
    print("Latency Metrics:")
    print(Latency)
    print("="*60)
    for key, values in Latency.items():
        if values:
            avg_latency = sum(values) / len(values)
            print(f"{key}: {avg_latency:.2f} seconds (average over {len(values)} runs)")
        else:
            print(f"{key}: No data available")



    


    