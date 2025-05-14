from collections import defaultdict
from evaluator import GPQAEvaluator
from generator import Generator, load_vLLM_model, generate_with_vLLM_model, LLM
from prompt import rag_prompt, eval_prompt
from generator import retriever
import numpy as np
from cfg import cfg
from huggingface_hub import login
from datasets import load_dataset, Dataset
from verify import probability_subanswer_question, probability_subquestion_question
import torch
import time
import random
from evaluate import run_evaluation
login('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')

model_name = "Qwen/Qwen2.5-7B-Instruct"
global global_value_model, global_tokenizer
model,tokenizer=LLM(model='qwen2.5')
global_value_model = model       
global_tokenizer = tokenizer
evaluator = GPQAEvaluator()
#tokenizer, model = load_vLLM_model(cfg.model_ckpt, cfg.seed, cfg.tensor_parallel_size, cfg.half_precision)
generator = Generator(cfg, tokenizer, model, evaluator)
ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
split = 100
df = ds['train'].select(range(split)).to_pandas()
# 변수설정
#reranker = True
reranker = False
rag_only_one = False
critic = False
output_list = []
input_list = []
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
    
    return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  # retriever.search_document_demo("What is the relationship between the lifetime of a quantum state and its energy uncertainty?",1)
# for i in range(len(ds['train']['Question'])):
  with torch.no_grad():
    global_value_model = model 
    global_tokenizer = tokenizer
    for row in range(split):
      question = ds['train']['Question'][row]
      prompt = ds['train'][row]['Question'] + 'A)' + ds['train'][row]['Correct Answer'].replace('\n','') + ' B)' + ds['train'][row]['Incorrect Answer 1'].replace('\n','') + ' C)' + ds['train'][row]['Incorrect Answer 2'].replace('\n','') + ' D)' + ds['train'][row]['Incorrect Answer 3'].replace('\n','')
      input_list.append(prompt)
      value_list = []
      final_questions = []
      subquestions_retrieval=[]
      best_subquestion = None  # 또는 ""로 초기화

      rag_score_dict = defaultdict()
    #-------------------subquestion------------------
      subquestions = generator.generate_subquestions(question,'')
      print('------------------------subquestions sampled----------------------------')
      for i in subquestions:
        value = probability_subquestion_question(question, subquestions) # probablistic score
        value_list.append(value)
      top_indices = np.argsort(value_list)[::-1][:args.num_subquestion]
      top3_subquestions = [subquestions[i] for i in top_indices] #top3 subquestions
      last_indices = np.argsort(value_list)[::-1][args.num_subquestion:]
      bad_subquestions = [subquestions[i] for i in last_indices]
    #--------------------retrieval-----------------
      for subquestion in top3_subquestions:
        best_score = float('-inf')
        if reranker: #reranker
          시retrieved_documents = retriever.search_document_demo(subquestion, 3)
          for retrieved_document in retrieved_documents:
            # score = llm_proposal(eval_prompt.format(rag_prompt.format(retrieved_document,subquestion)))
            score = generator.score(rag_prompt.format(context=retrieved_document['text'],question=subquestion))
            print(f'score: {score}')
            if score == 'not scored':
               pass
            if isinstance(score, float) and score > best_score:
              best_score = score
              best_subquestion = subquestion
              best_subquestion_retrieval = retrieved_document['text']
          if best_subquestion:
              subquestions_retrieval.append(rag_prompt.format(context=best_subquestion_retrieval,question=best_subquestion))
              print('best_subquestion succefully ranked')
          else:
              # subquestions_retrieval.append(rag_prompt.format(context=subquestion,question=retrieved_documents[0]['text']),'score가 점수아님')
              
              j = random.choice([0, 1, 2]) 
              subquestions_retrieval.append(rag_prompt.format(context=retrieved_documents[j]['text'], question=subquestion))
              print('best_subquestion not ranked')
        
        else: #reranker 아닐
          retrieved_documents = retriever.search_document_demo(subquestion, 1)
          subquestions_retrieval.append(rag_prompt.format(context=retrieved_documents[0]['text'], question=subquestion))
        print('------------------------한개 서브질문 with retrieval evaluated----------------------------')
        
          

    #------------------subanswer------------------------
      #for subquestion in subquestions_retrieval:
      io_output_list, subquestion_list, self_consistency_subanswer_list, value_list = generator.subanswer('',subquestions_retrieval,args.num_subanswer)
      print(f"subquestion에 대한 sc-answer: {self_consistency_subanswer_list}")
      print('------------------------one subanswer sampled----------------------------')        
        # probability_subanswer_question(ori_query, answer, ans_weight=0.75): 
    #-------------------final_output--------------------
      context = retriever.search_document_demo(subquestion, 1)[0]['text']
      output, only_answer = generator.final_output(prompt, top3_subquestions,self_consistency_subanswer_list,context)
      print('------------------------final_output_generated----------------------------')
      print(f'question:{question}')
      #torch.cuda.empty_cache()
      print(f'only answer (sc):{only_answer}')
      output_list.append(only_answer)
      print('--------------------------질문 한개 done---------------------------------')
    print(f'output_list:{output_list}')      
    print('------------------------evaluation 시작----------------------------')
    run_evaluation(df,input_list,output_list, output_dir="/abr/coss39/rag_scale/output")
# def run_evaluation(df, input_list, output_list,start_index=0, dataset_name='gpqa', output_dir='/content/output', split=1, apply_backoff=False):





  
