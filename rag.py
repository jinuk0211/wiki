
from generator import Generator, load_vLLM_model, generate_with_vLLM_model, llm_proposal, LLM
from prompt import rag_prompt
from generator import retriever,retriever2
from cfg import cfg
from huggingface_hub import login
from datasets import load_dataset, Dataset
import torch
import time
from evaluate import run_evaluation
from langchain_community.retrievers import WikipediaRetriever
import pandas as pd
#retriever = WikipediaRetriever()

#tokenizer, model = load_vLLM_model(cfg.model_ckpt, cfg.seed, cfg.tensor_parallel_size, cfg.half_precision)
model, tokenizer = LLM('qwen2.5')
ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
split = 100
df = ds['train'].select(range(split)).to_pandas()
output_list = []
input_list = []
rag = []
if __name__ == "__main__":
 
  with torch.no_grad():
    for row in range(split):
        
        question = ds['train']['Question'][row]
        prompt = ds['train'][row]['Question'] + 'A)' + ds['train'][row]['Correct Answer'].replace('\n','') + ' B)' + ds['train'][row]['Incorrect Answer 1'].replace('\n','') + ' C)' + ds['train'][row]['Incorrect Answer 2'].replace('\n','') + ' D)' + ds['train'][row]['Incorrect Answer 3'].replace('\n','')
           
        input_list.append(question)
        retrieved_document = retriever.search_document_demo(question, 1)    
        text = retrieved_document[0]['text']
        rag.append({"index": row, "text": text})


        #response = llm_proposal(model,tokenizer, rag_prompt.format(docs[0].page_content[:400],question=prompt))
        response = llm_proposal(model,tokenizer, rag_prompt.format(context=retrieved_document[0]['text'],question=prompt))
        print(response)
        final_input =  "Question: " + prompt + "\nAnswer: " + response[0] + '\n\n' + 'Extract only the final answer, without restating the question or explanation. Provide the answer as a short, direct response.'
        only_answer = llm_proposal(model,tokenizer,final_input)[0]
        output_list.append(only_answer)
    torch.cuda.empty_cache()
    print(output_list)
    df = pd.DataFrame(rag)
    df.to_csv("retrieved_texts.csv", index=False, encoding="utf-8")
    # evaluator.evaluate(only_answer, ds['train']['Correct Answer'][i])
    run_evaluation(df,input_list,output_list, output_dir='/abr/coss39/rag_scale/output/')


            

