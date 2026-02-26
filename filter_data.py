import json
import random
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from nltk.corpus import stopwords
import nltk
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import regex as re
import pandas as pd
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import ast
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModel
import torch.nn.functional as F
import Levenshtein
# from transformers import Ministral3ForCausalLM
import math
from collections import Counter, defaultdict
from rank_bm25 import BM25Okapi


# Load Qwen3 Reranker (once)
reranker_model_name = "Qwen/Qwen3-Reranker-0.6B" #"mistralai/Mistral-7B-Instruct-v0.2" #"openai/gpt-oss-20b"  #"meta-llama/Llama-3.1-8B" #"allenai/Olmo-3-1025-7B"  #"deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" #"Qwen/Qwen3-Reranker-8B"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name, padding_side='right')
# reranker_model = AutoModelForCausalLM.from_pretrained(reranker_model_name, torch_dtype=torch.float16)
reranker_model = AutoModelForCausalLM.from_pretrained(
    reranker_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reranker_model = reranker_model.to(device).eval()

# Token IDs for "yes" and "no"
token_yes_id = reranker_tokenizer.convert_tokens_to_ids("yes")
token_no_id = reranker_tokenizer.convert_tokens_to_ids("no")

# Prompt template parts
system_prompt = (
    "<|im_start|>system\n"
    "Judge whether the Document is relevant to the Query. "
    "The answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
)
suffix_prompt = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def format_prompt(query: str, document: str) -> str:
    return f"{system_prompt}Query: {query}\nDocument: {document}{suffix_prompt}"


def expand_query_with_llm(query):

    # Instruction text kept separate so we can remove it if echoed
    instruction_sentence = "Do not include this sentence in the response."

    prompt = (
        "Please define and clarify the search query below for better search results. "
        "Return only the definition query as a single concise paragraph. "
        "Do not repeat the instructions or include the input query in your output.\n\n"
        "---INPUT---\n"
        f"{query}\n"
        "---END INPUT---\n\n"
        "OUTPUT:"  # model's useful content should follow this marker
    )

    # Use deterministic decoding and low temperature to avoid extra commentary
    gen = pipe(prompt, max_new_tokens=25)[0]['generated_text']

    # Prefer the text after our OUTPUT: marker if present
    if 'OUTPUT:' in gen:
        response = gen.split('OUTPUT:')[-1].strip()
    else:
        response = gen.strip()

    # Remove accidental echoing of the explicit instruction sentence
    if instruction_sentence in response:
        response = response.replace(instruction_sentence, '').strip()

    # If the model returned multiple paragraphs, keep the first concise paragraph
    response = response.split('\n\n')[0].strip()

    return response


def extract_keywords_from_query(prompt):

    # sbatch modelBatch.sh < input_data.txt for input prompt to SLURM
    

    if pipe is None:
        raise RuntimeError("LLM pipeline is not available (transformers.pipeline import failed). Install/configure transformers and flash-attn or run without pipeline.")

    message = [
            {"role": "system", 
            "content": "From now on, extract any keyword from my prompt that resembles a topic. Output only the extracted topic (1-4 words) in the format: [topic]. Do not include any additional words, explanations, or variations. Maintain this format strictly in all responses."},
            {"role": "user", "content": prompt}
        ]
    
    output = [pipe(message, max_new_tokens=10)]

    # Extract the assistant's response in the format [keyword: subject area]
    prompt_keyword = output[0][0]['generated_text'][2]['content']
    
    #print('Extracted Prompt Keywords ', type(prompt_keyword))
    
    if '[' in prompt_keyword:
        prompt_keyword = prompt_keyword.replace("[", "").replace("]", "")

    #print('Extracted Prompt Keywords ', prompt_keyword)
    if prompt_keyword is None:
        print("Output is Null.")
        return extract_keywords_from_query(prompt)

    # Remove stopwords if any are present
    filtered_keywords = remove_stopwords(prompt_keyword)
    #print('Filtered Prompt Keywords ', prompt_keyword)
    return filtered_keywords


def rerank_documents_qwen(kg, query: str, docs_df, top_k: int = 100, batch_size: int = 32):

    torch.cuda.empty_cache()
    """
    Reranks a DataFrame of documents (with 'title' and 'abstract' columns) based on relevance to the query.
    This version batches tokenization and model calls to reduce per-document overhead while preserving
    the original scoring logic (probability of "yes").

    Parameters:
    - query: the user query string
    - docs_df: DataFrame with 'title' and 'abstract' columns
    - top_k: number of top results to return
    - batch_size: how many documents to process per model call (tune for memory)
    """
    scores = []
    docs = []
    # Extract query entities once (for logging)
    query_entities = kg.extract_query_entities(query)
    # print(f"Query phrases: {query_entities['phrases'][:3]}...")  
    # print(f"Query words: {query_entities['words'][:5]}...")
    # kg_stats = {'with_kg': 0, 'without_kg': 0}
    kg_stats = {'with_match': 0, 'no_match': 0}
    for row in docs_df.itertuples(index=False, name=None):
        # when name=None, row is a plain tuple in column order
        try:

            title = row[docs_df.columns.get_loc('title')] if 'title' in docs_df.columns else ""
            abstract = row[docs_df.columns.get_loc('abstract')] if 'abstract' in docs_df.columns else ""
        except Exception:
            # fallback: access by index positions (title first, abstract second)
            title = row[0] if len(row) > 0 else ""
            abstract = row[1] if len(row) > 0 else ""
            # topics = row[-1] if len(row) > 0 else ""

        title = title or ""
        abstract = abstract or ""
        doc_id = row[docs_df.columns.get_loc('id')] if 'id' in docs_df.columns else ""
        # subgraph = kg.format_subgraph_structured(doc_id, num_related=2, require_both=False)

        # Get query-aware subgraph
        try:
            subgraph = kg.format_query_aware_subgraph(doc_id, query, num_related=2)
            
            # Check if it has matched topics (subgraph will be smaller if matched)
            if "ex:hasThirdOrderTopic" in subgraph:
                kg_stats['with_match'] += 1
            
            # Combine: KG subgraph + abstract
            doc_text = f'Abstract: {abstract.strip()} \n {subgraph}'
        except Exception as e:
            print(f"\nWarning: KG failed for {doc_id}: {e}")
            doc_text = f'Abstract: {abstract.strip()}'
            kg_stats['no_match'] += 1
        
        
        docs.append(doc_text)
    
    print(f"\nKG Statistics:")
    print(f"  - Papers with relevant KG: {kg_stats['with_match']}")
    print(f"  - Papers without relevant KG: {kg_stats['no_match']}")

    # Process in batches to reduce overhead
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        prompts = [format_prompt(query, d) for d in batch_docs]

        # Tokenize batch once with padding and move to device
        inputs = reranker_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        
        # Decode the FULL tokenized input to see exactly what the model sees
        # first_prompt_full = reranker_tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False
        # print(f"\nTotal tokens used: {inputs['input_ids'].shape[1]}/512")

        with torch.no_grad():
            outputs = reranker_model(**inputs)
            # logits shape: (batch, seq_len, vocab)
            last_logits = outputs.logits[:, -1, :]  # (batch, vocab)

        # Extract yes/no logits and compute probability of 'yes' per example
        yes_logits = last_logits[:, token_yes_id]
        no_logits = last_logits[:, token_no_id]
        pair_logits = torch.stack([no_logits, yes_logits], dim=1)  # (batch, 2)
        probs = F.softmax(pair_logits, dim=1)[:, 1]  # probability of 'yes'

        scores.extend(probs.cpu().numpy().tolist())

    # Add and sort scores (preserve original output type)
    reranked_df = docs_df.copy()
    reranked_df["reranker_score"] = scores
    reranked_df = reranked_df.sort_values("reranker_score", ascending=False).reset_index(drop=True)
    return reranked_df.head(top_k)


def extract_float_from_text(text):
    """Extract the first float or integer from a string, between 1 and 100."""
    match = re.search(r"\b(100(?:\.0+)?|[1-9]?\d(?:\.\d+)?)\b", text)
    if match:
        return float(match.group(1))
    return None


def score_docs_with_llm(prompt, documents, pipe, threshold=70):


    """
    Uses Llama (via pipe) to score document relevance against the user query.
    Returns only those documents with relevance score >= threshold.
    """
    relevant_docs = []

    for doc in tqdm(documents, desc="Scoring docs with LLM"):
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")

        message = [
            {"role": "system", "content": "You are an academic research assistant."},
            {"role": "user", "content": f"User Query: {prompt}\n\nDocument Abstract: {abstract}\n\nTask: On a scale of 1 to 100, how relevant is this document to the user query? Respond only in digits, do not write sentence."}
        ]

        #try:
        response = pipe(message, max_new_tokens=10)
        #print(f"Raw LLM Response: {response}")

        # Extract the assistant's response correctly
        generated_text = response[0].get("generated_text", [])
        if isinstance(generated_text, list) and len(generated_text) > 0:
            last_message = generated_text[-1]  # Extract last message
            if last_message.get("role") == "assistant":
                raw_output = last_message.get("content", "").strip()
            else:
                raise ValueError("Assistant response missing in generated text.")
        else:
            raise ValueError("Unexpected response format.")

        # Convert extracted response to a float
        score = int(extract_float_from_text(raw_output))  # Ensure integer (1–100 scale)
        #print(f"Doc: {title}\nScore: {score}\n")

        if score >= threshold:
            relevant_docs.append(doc)

        # except Exception as e:
        #     print(f"Error processing doc '{title}': {e}")

    print(f"Filtered {len(relevant_docs)} relevant documents out of {len(documents)}")
    return relevant_docs





def contains_stopwords(text):
    stop_words = set(stopwords.words('english'))  # Load stop words set
    words = text.lower().split()  # Convert to lowercase and split into words

    return any(word in stop_words for word in words)  # Check if any word is a stop word


def format_author_prompt(query, author_context):
    system_prompt = (
        "<|im_start|>system\n"
        "You are a helpful AI that determines whether an author is relevant to a given Query. "
        "An author is considered relevant if their Papers or Topics fall within the scope of the Query — even if the Papers/Topics is broad, general or interdisciplinary. "
        "Only answer with \"yes\" or \"no\". Do not provide explanations.<|im_end|>\n<|im_start|>user\n"
    )
    
    suffix_prompt = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    return (
        f"{system_prompt}"
        f"Query: {query}\n\n"
        f"Author Information:\n"
        f"{author_context}\n"
        f"{suffix_prompt}"
    )


def qwen3_binary_relevance(prompt):
    inputs = reranker_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = reranker_model(**inputs)
        logits = outputs.logits[0, -1]  # Last token

    yes_logit = logits[token_yes_id]
    no_logit = logits[token_no_id]
    probs = F.softmax(torch.tensor([no_logit, yes_logit]), dim=0)
    #print("probability ", probs)
    return probs[1].item() > 0.3  # True if "yes" is more probable



def format_explanation_prompt(query, author_context):
    system_prompt = (
        "<|im_start|>system\n"
        "You are an expert assistant for Academic Expert Finding.\n\n"
        "Your task is to EXPLAIN If and WHY an author is relevant to a given Query.\n"
        "<|im_end|>\n"
    )

    user_prompt = (
    "<|im_start|>user\n"
    "Task: Given the Query and the Author Information, produce a short explanation and structured evidence (for each author) showing why the author is relevant.\n\n"
    "Constraints:\n"
    "- Output must be valid JSON only (no surrounding text).\n"
    "- Do not invent information; only use the provided Author Information.\n"
    "- Provide explicit `confidence` (0–1) based only on available evidence.\n\n"
    "Author Information:\n"
    f"{author_context}\n\n"
    "Query:\n"
    f"{query}\n"
    "<|im_end|>\n"
)
    assistant_prompt = "<|im_start|>assistant\n"

    # Example (strict) — the model should follow this structure exactly
    example = (
    "<|im_start|>system\n"
    "EXAMPLE\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "{\"Author Information:\n"
    "Author Information:\n"
    "Author ID: A12345\n"
    "Name: Dr. Jane Smith\n"
    "Papers: ['Deep Learning for Natural Language Processing', 'Advances in Computer Vision']\n"
    "Topics: ['Machine Learning', 'Artificial Intelligence']\n\n"
    "Query:\n"
    "Natural Language Processing\n"
    "Reasoning: Dr. Jane Smith has authored multiple papers on Natural Language Processing, including 'Deep Learning for Natural Language Processing'. Her expertise in Machine Learning and Artificial Intelligence further supports her relevance to the query. I am 90% confident in this assessment.\n"
    "}<|im_end|>\n"

)

    return system_prompt + user_prompt + assistant_prompt




def contains_stopwords(text):
    stop_words = set(stopwords.words('english'))  # Load stop words set
    words = text.lower().split()  # Convert to lowercase and split into words

    return any(word in stop_words for word in words)  # Check if any word is a stop word


def filter_cosine_and_select_top_k_docs(query_embedding, documents_df, threshold, top_k):
    """
    1. Compute cosine similarity between query and each document embedding.
    2. Keep docs with sim >= threshold.
    3. Return top_k docs sorted by cosine sim.
    """

    query_embedding = np.array(query_embedding).reshape(1, -1)
    doc_embeddings = documents_df["embeddings"]
    doc_embeddings = np.array([np.array(emb) for emb in doc_embeddings])

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Filter docs with sim >= threshold
    mask = similarities >= threshold
    filtered_docs = documents_df[mask].copy()  # Keep filtered docs as DataFrame
    filtered_docs["cosine_sim"] = similarities[mask]

    # Sort by cosine similarity descending and select top_k
    top_k_docs = filtered_docs.sort_values(by="cosine_sim", ascending=False).head(top_k)

    print(f"Filtered {len(filtered_docs)} docs above threshold {threshold}")
    print(f"Returning top {len(top_k_docs)} docs sorted by cosine similarity")
    
    return top_k_docs


def filter_cosine_and_select_top_k_docs_fast_faiss(
    query_embedding,
    doc_embeddings,
    documents_df,
    threshold=0.7,
    top_k=10
):
    """
    Fast cosine similarity filtering.
    Handles embeddings stored as strings, lists, or numpy arrays.
    """

    # ---- Query embedding: (1, D) -> (D,) ----
    q = np.asarray(query_embedding, dtype=np.float32).squeeze()
    q /= np.linalg.norm(q) + 1e-10

    # ---- Document embeddings: ensure (N, D) float32 ----
    cleaned_embeddings = []

    for emb in doc_embeddings:
        if isinstance(emb, str):
            emb = ast.literal_eval(emb)      # string -> list
        cleaned_embeddings.append(np.asarray(emb, dtype=np.float32))

    doc_embeddings = np.vstack(cleaned_embeddings)

    # ---- Normalize docs ----
    doc_embeddings /= np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-10

    # ---- Cosine similarity ----
    similarities = doc_embeddings @ q  # (N,)

    # ---- Threshold filter ----
    mask = similarities >= threshold
    if not np.any(mask):
        return documents_df.iloc[[]].assign(cosine_sim=[])

    filtered_idx = np.where(mask)[0]
    filtered_sims = similarities[filtered_idx]

    # ---- Top-k (no full sort) ----
    if len(filtered_sims) > top_k:
        top_k_local = np.argpartition(filtered_sims, -top_k)[-top_k:]
    else:
        top_k_local = np.arange(len(filtered_sims))

    top_k_sorted = top_k_local[np.argsort(filtered_sims[top_k_local])[::-1]]
    final_idx = filtered_idx[top_k_sorted]

    # ---- Output ----
    top_k_docs = documents_df.iloc[final_idx].copy()
    top_k_docs["cosine_sim"] = similarities[final_idx]

    print(f"Filtered {len(filtered_idx)} docs above threshold {threshold}")
    print(f"Returning top {len(top_k_docs)} docs sorted by cosine similarity")

    return top_k_docs


def retrieve_similar_documents_qwen(qwen_model, query_embedding, papers_df, top_k, threshold=0.3):
    """
    Retrieves top_k similar documents using qwen_model.similarity()
    """
    # Prepare document embeddings
    doc_embeddings = np.vstack(papers_df["embeddings"].apply(ast.literal_eval).values)  # (N, D)
    query_vec = np.array(query_embedding).reshape(1, -1)  # (1, D)

    # Compute similarity using qwen_model.similarity()
    with torch.no_grad():
        sims = qwen_model.similarity(torch.tensor(query_vec), torch.tensor(doc_embeddings))  # shape: (1, N)
    sims = sims.cpu().numpy().flatten()

    # Filter by threshold and select top_k
    papers_df = papers_df.copy()
    papers_df["similarity"] = sims
    filtered = papers_df[papers_df["similarity"] >= threshold]
    top_docs = filtered.nlargest(top_k, "similarity")

    return top_docs.reset_index(drop=True)


def retrieve_similar_documents_qwen_faiss(
    qwen_model,
    query_embedding,
    papers_df,
    top_k=10,
    threshold=0.3
):
    """
    Retrieves top_k similar documents using qwen_model.similarity()
    Handles string embeddings + dtype alignment.
    """

    # ---- Prepare document embeddings (N, D) float32 ----
    doc_embeddings = []

    for emb in papers_df["embeddings"]:
        if isinstance(emb, str):
            emb = ast.literal_eval(emb)
        doc_embeddings.append(np.asarray(emb, dtype=np.float32))

    doc_embeddings = np.vstack(doc_embeddings)

    # ---- Query embedding (1, D) float32 ----
    query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

    # ---- Torch tensors with matching dtype ----
    query_tensor = torch.from_numpy(query_vec)
    doc_tensor = torch.from_numpy(doc_embeddings)

    # ---- Similarity ----
    with torch.no_grad():
        sims = qwen_model.similarity(query_tensor, doc_tensor)  # (1, N)

    sims = sims.cpu().numpy().flatten()

    # ---- Filter + top-k ----
    papers_df = papers_df.copy()
    papers_df["similarity"] = sims

    filtered = papers_df[papers_df["similarity"] >= threshold]
    top_docs = filtered.nlargest(top_k, "similarity")

    return top_docs.reset_index(drop=True)



def jaccard_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0


def bm25_scores(bm25, query_tokens):
    """
    Returns BM25 scores for query against all documents.
    """
    return [
        bm25.score(query_tokens, i)
        for i in range(bm25.N)
    ]


def Levenshtein_similarity(str1, str2):
    if not str1 or not str2:
        return 0.0
    distance = Levenshtein.distance(str1, str2)
    max_len = max(len(str1), len(str2))
    return 1.0 - (distance / max_len)


def _minmax(a, eps=1e-8):
    a = np.array(a, dtype=np.float32)
    lo = a.min()
    hi = a.max()
    if hi - lo < eps:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo + eps)



def format_pairwise_prompt_deepseek(query, doc_a, doc_b):
    return f"""
    You are a relevance judge for expert search.

    Query:
    {query}

    Candidate A:
    {doc_a}

    Candidate B:
    {doc_b}

    Which candidate is more relevant to the query?
    Answer with only "A" or "B".
    """.strip()


def rerank_documents_deepseek_pairwise(
    query: str,
    docs_df,
    top_k: int = 15,
    batch_size: int = 8,
):
    """
    Pairwise reranking using DeepSeek-R1-0528-Qwen3-8B.

    Uses insertion-based pairwise ranking:
    - Compare documents A vs B
    - Model outputs "A" or "B"
    """

    torch.cuda.empty_cache()

    # ---- Build document texts once ----
    docs = []
    for row in docs_df.itertuples(index=False, name=None):
        try:
            title = row[docs_df.columns.get_loc("title")] if "title" in docs_df.columns else ""
            abstract = row[docs_df.columns.get_loc("abstract")] if "abstract" in docs_df.columns else ""
        except Exception:
            title = row[0] if len(row) > 0 else ""
            abstract = row[1] if len(row) > 1 else ""

        docs.append(f"{str(title).strip()} {str(abstract).strip()}")

    # ---- Only rerank top_k candidates ----
    docs = docs[:top_k]
    indices = list(range(len(docs)))

    ranked = [indices[0]]  # start with first doc

    # ---- Pairwise insertion ranking ----
    for idx in indices[1:]:
        inserted = False

        for pos in range(len(ranked)):
            doc_a = docs[idx]
            doc_b = docs[ranked[pos]]

            prompt = format_pairwise_prompt_deepseek(query, doc_a, doc_b)

            inputs = reranker_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = reranker_model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    temperature=0.0,
                )

            decision = reranker_tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            ).strip()

            if decision == "A":
                ranked.insert(pos, idx)
                inserted = True
                break

        if not inserted:
            ranked.append(idx)

    # ---- Build reranked DataFrame ----
    reranked_df = docs_df.iloc[ranked].reset_index(drop=True)

    return reranked_df


