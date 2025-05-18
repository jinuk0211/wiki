from collections import defaultdict
from evaluator import GPQAEvaluator
from generator import Generator, load_vLLM_model, generate_with_vLLM_model, LLM
from prompt import rag_prompt, eval_prompt
#from generator import \ever
import numpy as np
from cfg import cfg
from huggingface_hub import login
from datasets import load_dataset, Dataset
from verify import probability_subanswer_question, probability_subquestion_question
import torch
import time
import random
from evaluate import run_evaluation
import pandas as pd
login('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')

model_name = "Qwen/Qwen2.5-7B-Instruct"
global global_value_model, global_tokenizer
model,tokenizer=LLM(model='qwen2.5')
global_value_model = model       
global_tokenizer = tokenizer
evaluator = GPQAEvaluator()
#tokenizer, model = load_vLLM_model(cfg.model_ckpt, cfg.seed, cfg.tensor_parallel_size, cfg.half_precision)
generator = Generator(cfg, tokenizer, model, evaluator)

split = 100


#reranker = True
reranker = False
rag_only_one = False
critic = False
output_list = []
input_list = []
subquestions_list = []
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Subquestion generator script.")
    
    parser.add_argument(
        '--num_subquestion',
        type=int,
        default=2,
        help='Number of subquestions to generate per main question (default: 3)'
    )
    parser.add_argument(
        '--num_subanswer',
        type=int,
        default=2,
        help='Number of subanswer to generate per main question (default: 3)'
    )    
    parser.add_argument(
        '--ds',
        type=str,
        default='popqa',
        help='name of data'
    )        
    return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  if args.ds == 'gpqa':
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    df = ds['train'].to_pandas()
  elif args.ds == 'popqa':
    df = pd.read_csv("hf://datasets/akariasai/PopQA/test.tsv", sep="\t")
    df = df[:200]
    print('indexed 200')
  elif args.ds == 'arc':
    splits = {'train': 'ARC-Challenge/train-00000-of-00001.parquet', 'test': 'ARC-Challenge/test-00000-of-00001.parquet', 'validation': 'ARC-Challenge/validation-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/allenai/ai2_arc/" + splits["train"])

    
  with torch.no_grad():
    global_value_model = model 
    global_tokenizer = tokenizer
    
    for row in range(len(df)):
      if args.ds == 'gpqa':
        question = ds['train']['Question'][row]
        prompt = ds['train'][row]['Question'] + 'A)' + ds['train'][row]['Correct Answer'].replace('\n','') + ' B)' + ds['train'][row]['Incorrect Answer 1'].replace('\n','') + ' C)' + ds['train'][row]['Incorrect Answer 2'].replace('\n','') + ' D)' + ds['train'][row]['Incorrect Answer 3'].replace('\n','')
      elif args.ds == 'popqa':
        question = df['question'][row]
        prompt = question 
      elif args.ds == 'arc':
        question = df['question'][row]
        prompt = question + "\nA)" + df['choices'][row]['text'][0] + " B)" + df['choices'][row]['text'][1] + " C)" + df['choices'][row]['text'][2] + " D)" + df['choices'][row]['text'][3]
      
      
      
      input_list.append(prompt)
      value_list = []
      final_questions = []
      subquestions_retrieval=[]
      best_subquestion = None  

      rag_score_dict = defaultdict()
    
      subquestions = generator.generate_subquestions(question,'')
      #if len(subquestions) < 2:
        #rephrased_subq = generator.rephrase(subquestions[0])
        #subquestions.append(rephrased_subq)
      print(f'number of subquestions:{len(subquestions)} \n contents of subquestions:{subquestions}')
      for i in subquestions:
        value = probability_subquestion_question(question, subquestions) # probablistic score
        value_list.append(value)
        print(f'values of subquestions{value_list}')
      top_indices = np.argsort(value_list)[::-1][:args.num_subquestion]
      top_subquestions = [subquestions[i] for i in top_indices] #top3 subquestions
      last_indices = np.argsort(value_list)[::-1][args.num_subquestion:]
      bad_subquestions = [subquestions[i] for i in last_indices]
      if len(subquestions) > 1:
        subquestions_list.append({"index": row, "subquestion1": top_subquestions[0],"subquestion2":top_subquestions[1]})
      else:
        
        subquestions_list.append({"index": row, "subquestion1": top_subquestions[0],"subquestion2": None})
    df = pd.DataFrame(subquestions_list)
    df.to_csv(f'subquestions_{args.ds}.csv', index=False, encoding="utf-8")         