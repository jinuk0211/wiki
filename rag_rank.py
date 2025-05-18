from collections import defaultdict
from evaluator import GPQAEvaluator
from generator import Generator, load_vLLM_model, generate_with_vLLM_model, LLM, llm_proposal
from prompt import rag_prompt, eval_prompt
import numpy as np
from cfg import cfg
from huggingface_hub import login
from datasets import load_dataset, Dataset
from verify import probability_subanswer_question, probability_subquestion_question, probability_context_question
import torch
import time
import random
from evaluate import run_evaluation
import pandas as pd
login('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')
import argparse
from langchain_community.retrievers import WikipediaRetriever
model_name = "Qwen/Qwen2.5-7B-Instruct"
global global_value_model, global_tokenizer
model,tokenizer=LLM(model='qwen2.5')
global_value_model = model       
global_tokenizer = tokenizer
evaluator = GPQAEvaluator()
generator = Generator(cfg, tokenizer, model, evaluator)
ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
split = 100
#df = ds['train'].to_pandas()
total_len = len(ds['train'])
df = ds['train'].select(range(100, total_len)).to_pandas()
input_list = []
output_list = []
subquestions_list = []
rag_df = pd.read_csv('retrieved_texts.csv')
subquestions_df = pd.read_csv("subquestions.csv")
naive_rag = True

def parse_args():
    parser = argparse.ArgumentParser(description="Subquestion generator script.")
    
    parser.add_argument(
        '--naive_rag',
        action='store_true',
        help='Use naive RAG mode (flag). Default: False'
    )

    parser.add_argument(
        '--ds',
        type=str,
        default='popqa',
        help='name of data'
    )        
    return parser.parse_args()   
     
if __name__ == '__main__':
  args = parse_args()
  if args.ds == 'gpqa':
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
        df = ds['train'].to_pandas()
      elif args.ds == 'popqa':
        df = pd.read_csv("hf://datasets/akariasai/PopQA/test.tsv", sep="\t")
      elif args.ds == 'arc':
        splits = {'train': 'ARC-Challenge/train-00000-of-00001.parquet', 'test': 'ARC-Challenge/test-00000-of-00001.parquet', 'validation': 'ARC-Challenge/validation-00000-of-00001.parquet'}
        df = pd.read_parquet("hf://datasets/allenai/ai2_arc/" + splits["train"])
      elif args.ds == 'musique':
        ds = load_dataset("dgslibisey/MuSiQue")
            
  for row in range(len(ds['train'])):
    retrived_texts = []
    subquestions_retrieval=[]
    value_list = []
    
    if args.ds == 'gpqa':
      question = ds['train']['Question'][row]
      prompt = ds['train'][row]['Question'] + 'A)' + ds['train'][row]['Correct Answer'].replace('\n','') + ' B)' + ds['train'][row]['Incorrect Answer 1'].replace('\n','') + ' C)' + ds['train'][row]['Incorrect Answer 2'].replace('\n','') + ' D)' + ds['train'][row]['Incorrect Answer 3'].replace('\n','')
      
    elif args.ds == 'popqa':
      question = df['question'][row]
      prompt = question
       
    elif args.ds == 'arc':
      question = df['question'][row]
      prompt = question + "\nA)" + df['choices'][row]['text'][0] + " B)" + df['choices'][row]['text'][1] + " C)" + df['choices'][row]['text'][2] + " D)" + df['choices'][row]['text'][3]       
      
          
    retrieved_texts=rag_df.loc[row, ["text", "text2", "text3"]].tolist()
    
    input_list.append(question)
    
           
    for retrieved_text in retrieved_texts:
      value = probability_context_question(question, retrieved_text) # probablistic score
      value_list.append(value)
      
    top_index = np.argmax(value_list)
    
    top_retrieved_text = retrieved_texts[top_index]
    response = llm_proposal(model,tokenizer, rag_prompt.format(context=top_retrieved_text,question=prompt))
    print(f'question{row} answer: {response}')
    final_input =  "Question: " + prompt + "\nAnswer: " + response[0] + '\n\n' + 'Extract only the final answer, without restating the question or explanation. Provide the answer as a short, direct response.'
    only_answer = llm_proposal(model,tokenizer,final_input)[0]
    output_list.append(only_answer)
    torch.cuda.empty_cache()
      

  print(f'output_list:{output_list}')         
  print(output_list)
  
    
  run_evaluation(df,input_list,output_list, output_dir='/abr/coss39/rag_scale/output/')    