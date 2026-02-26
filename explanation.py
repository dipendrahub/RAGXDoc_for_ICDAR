from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import List, Dict



class ExplainableReranker:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        """Load model once"""
        print(f"Loading {model_name}...")
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded!")
    
    def generate_explanation_and_rerank(self, query: str, paper_id: str, title: str, 
                                       abstract: str, subgraph: str, preranker_score: float) -> dict:
        """
        Generate explanation AND get confidence score for reranking
        
        Returns:
            {
                "paper_id": str,
                "title": str,
                "query": str,
                "preranker_score": float,
                "explanation": {
                    "relevance": "yes" or "no",
                    "confidence": "high", "medium", or "low",
                    "decision_summary": str,
                    "topic_alignment": {"matched_topics": list},
                    "key_evidence": list
                },
                "final_rank_score": float  # For sorting
            }
        """
        
        # Enhanced prompt with confidence
        prompt = f"""Query: {query}

Paper Title: {title}

Topics from Knowledge Graph:
{subgraph}

Abstract: {abstract[:600]}

Pre-ranker Score: {preranker_score:.4f}

Task: Evaluate this paper's relevance to the query and explain your decision.

Output in this exact format:
RELEVANCE: [yes or no]
CONFIDENCE: [high, medium, or low]
DECISION: [1-2 sentences explaining relevance]
TOPICS: [comma-separated list of relevant topics]
EVIDENCE: [quote from abstract] | [another quote if needed]

Your response:"""

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        if "Your response:" in response:
            answer = response.split("Your response:")[-1].strip()
        else:
            answer = response
        
        # Parse response
        explanation = self._parse_response(answer, preranker_score)
        
        # Calculate final ranking score
        final_score = self._calculate_ranking_score(
            relevance=explanation['relevance'],
            confidence=explanation['confidence'],
            preranker_score=preranker_score
        )
        
        return {
            "paper_id": paper_id,
            "title": title,
            "query": query,
            "preranker_score": float(preranker_score),
            "explanation": explanation,
            "final_rank_score": final_score
        }
    
    def _parse_response(self, text: str, preranker_score: float) -> dict:
        """Parse model response"""
        
        relevance = "yes"
        confidence = "medium"
        decision = ""
        topics = []
        evidence = []
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("RELEVANCE:"):
                rel_text = line.replace("RELEVANCE:", "").strip().lower()
                relevance = "no" if "no" in rel_text else "yes"
            
            elif line.startswith("CONFIDENCE:"):
                conf_text = line.replace("CONFIDENCE:", "").strip().lower()
                if "high" in conf_text:
                    confidence = "high"
                elif "low" in conf_text:
                    confidence = "low"
                else:
                    confidence = "medium"
            
            elif line.startswith("DECISION:"):
                decision = line.replace("DECISION:", "").strip()
            
            elif line.startswith("TOPICS:"):
                topics_str = line.replace("TOPICS:", "").strip()
                topics = [t.strip() for t in topics_str.split(',') if t.strip()]
            
            elif line.startswith("EVIDENCE:"):
                evidence_str = line.replace("EVIDENCE:", "").strip()
                evidence = [e.strip() for e in evidence_str.split('|') if e.strip()]
        
        # Fallback
        if not decision:
            decision = text[:200] if text else "Could not generate explanation"
        
        # Infer relevance from preranker score if not provided
        if not any(line.startswith("RELEVANCE:") for line in lines):
            relevance = "yes" if preranker_score >= 0.5 else "no"
        
        return {
            "relevance": relevance,
            "confidence": confidence,
            "decision_summary": decision,
            "topic_alignment": {
                "matched_topics": topics
            },
            "key_evidence": evidence
        }
    
    def _calculate_ranking_score(self, relevance: str, confidence: str, preranker_score: float) -> float:
        """
        Calculate final ranking score based on relevance + confidence
        
        Ranking order:
        1. relevance=yes, confidence=high
        2. relevance=yes, confidence=medium
        3. relevance=yes, confidence=low
        4. relevance=no (all grouped at bottom)
        
        Returns:
            Score for sorting (higher = more relevant)
        """
        
        if relevance == "yes":
            if confidence == "high":
                base_score = 3.0
            elif confidence == "medium":
                base_score = 2.0
            else:  # low
                base_score = 1.0
        else:  # relevance == "no"
            base_score = 0.0
        
        # Add preranker score as tiebreaker (normalized to 0-1 range)
        final_score = base_score + (preranker_score * 0.9)  # Max contribution: 0.9
        
        return final_score
    
    def rerank_with_explanations(self, query: str, preranked_df, kg, top_k: int = 15) -> pd.DataFrame:
        """
        Rerank top-k papers from pre-ranker with LLM explanations
        
        Args:
            query: User query
            preranked_df: DataFrame from previous ranker with columns: id, title, abstract, reranker_score
            kg: KG extractor
            top_k: Number of papers to rerank
        
        Returns:
            DataFrame sorted by final_rank_score with explanations
        """
        results = []
        
        print(f"LLM RERANKING WITH EXPLANATIONS")
        print(f"Query: {query}")
        print(f"Processing top {top_k} papers from pre-ranker...\n")
        
        for idx, row in preranked_df.head(top_k).iterrows():
            paper_id = str(row['id'])
            # title = row.get('title', 'Unknown')
            title = str(row.get('title', '') or '')
            # abstract = row.get('abstract', '')
            title = str(row.get('abstract', '') or '')
            preranker_score = row.get('reranker_score', 0.0)
            
            # Get subgraph
            try:
                subgraph = kg.format_query_aware_subgraph(paper_id, query, num_related=2)
            except:
                subgraph = f"Paper: {title}"
            
            # Generate explanation and get ranking score
            print(f"  {idx+1}/{top_k} - {title[:60]}...")
            
            result = self.generate_explanation_and_rerank(
                query=query,
                paper_id=int(paper_id),
                title=title,
                abstract=abstract,
                subgraph=subgraph,
                preranker_score=preranker_score
            )
            
            results.append(result)
        
        # Convert to DataFrame
        reranked_df = pd.DataFrame(results)
        
        # Sort by final_rank_score (descending)
        reranked_df = reranked_df.sort_values('final_rank_score', ascending=False).reset_index(drop=True)
        
        # Print summary
        print(f"RERANKING SUMMARY")
        
        yes_high = len(reranked_df[(reranked_df['explanation'].apply(lambda x: x['relevance']) == 'yes') & 
                                   (reranked_df['explanation'].apply(lambda x: x['confidence']) == 'high')])
        yes_medium = len(reranked_df[(reranked_df['explanation'].apply(lambda x: x['relevance']) == 'yes') & 
                                     (reranked_df['explanation'].apply(lambda x: x['confidence']) == 'medium')])
        yes_low = len(reranked_df[(reranked_df['explanation'].apply(lambda x: x['relevance']) == 'yes') & 
                                  (reranked_df['explanation'].apply(lambda x: x['confidence']) == 'low')])
        no_count = len(reranked_df[reranked_df['explanation'].apply(lambda x: x['relevance']) == 'no'])
        
        print(f"Relevance=yes, Confidence=high:   {yes_high}")
        print(f"Relevance=yes, Confidence=medium: {yes_medium}")
        print(f"Relevance=yes, Confidence=low:    {yes_low}")
        print(f"Relevance=no:                     {no_count}")
        print(f"\nTop 10 should be highly relevant papers ✓")
        print(f"{'='*80}\n")
        
        return reranked_df





