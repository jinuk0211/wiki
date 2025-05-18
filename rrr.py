
from generator import Generator, load_vLLM_model, generate_with_vLLM_model, llm_proposal, LLM
from prompt import rag_prompt
from generator import retriever
from cfg import cfg
from huggingface_hub import login
from datasets import load_dataset, Dataset
import torch
import time
from evaluate import run_evaluation
import argparse

#tokenizer, model = load_vLLM_model(cfg.model_ckpt, cfg.seed, cfg.tensor_parallel_size, cfg.half_precision)
model, tokenizer = LLM('qwen2.5')
split = 10
output_list = []
input_list = []

def parse_args():
    parser = argparse.ArgumentParser(description="Subquestion generator script.")
    
    parser.add_argument(
        '--ds',
        type=str,
        default='popqa',
        help='name of data'
    )        
    return parser.parse_args()


def _parse(text):
    return text.strip('"').strip("**")
    
if __name__ == "__main__":

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    df = ds['train'].to_pandas()
  elif args.ds == 'popqa':
    df = pd.read_csv("hf://datasets/akariasai/PopQA/test.tsv", sep="\t")
  elif args.ds == 'arc':
    splits = {'train': 'ARC-Challenge/train-00000-of-00001.parquet', 'test': 'ARC-Challenge/test-00000-of-00001.parquet', 'validation': 'ARC-Challenge/validation-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/allenai/ai2_arc/" + splits["train"])
  elif args.ds == 'musique':
    ds = load_dataset("dgslibisey/MuSiQue")
    
    
  with torch.no_grad():
    for row in range(split):
        if args.ds == 'gpqa':
          question = ds['train']['Question'][row]
          
        elif args.ds == 'popqa':
          question = df['question'][row]
      
        elif args.ds == 'arc':
          question = df['question'][row]
          
        
        rrr_prompt= """Provide a better search query for web search engine to answer the given question, end the queries with ¡¯**¡¯. Question: {x} Answer:"""
        rrr_question = llm_proposal(model,tokenizer, rrr_prompt.format(x=question))
        input_list.append(rrr_question)
        
        retrieved_document = retriever.search_document_demo(rrr_question, 1)

        if args.ds == 'gpqa':
          prompt = rrr_question + 'A)' + ds['train'][row]['Correct Answer'].replace('\n','') + ' B)' + ds['train'][row]['Incorrect Answer 1'].replace('\n','') + ' C)' + ds['train'][row]['Incorrect Answer 2'].replace('\n','') + ' D)' + ds['train'][row]['Incorrect Answer 3'].replace('\n','')
        elif args.ds == 'popqa':
          prompt = rrr_question 
        elif args.ds == 'arc':
          prompt = rrr_question + "\nA)" + df['choices'][row]['text'][0] + " B)" + df['choices'][row]['text'][1] + " C)" + df['choices'][row]['text'][2] + " D)" + df['choices'][row]['text'][3]  
          
            
        response = llm_proposal(model,tokenizer, rag_prompt.format(context=retrieved_document[0]['text'],question=prompt))
        print(response)
        final_input = 'output:' + response[0] + "Therefore, the answer is "
        only_answer = llm_proposal(model,tokenizer,final_input)[0]
        output_list.append(only_answer)
    torch.cuda.empty_cache()
    print(output_list)
    # evaluator.evaluate(only_answer, ds['train']['Correct Answer'][i])
    run_evaluation(df,input_list,output_list, output_dir='/abr/coss39/rag_scale/output/')

"""Provide a better search query for \
web search engine to answer the given question, end \
the queries with ¡¯**¡¯. Question: \
{x} Answer:"""