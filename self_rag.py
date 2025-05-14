
from generator import Generator, load_vLLM_model, generate_with_vLLM_model, llm_proposal, LLM
from prompt import rag_prompt
from generator import retriever
from cfg import cfg
from huggingface_hub import login
from datasets import load_dataset, Dataset
import torch
import time
from evaluate import run_evaluation

#tokenizer, model = load_vLLM_model(cfg.model_ckpt, cfg.seed, cfg.tensor_parallel_size, cfg.half_precision)
model, tokenizer = LLM('qwen2.5')
ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
split = 5
df = ds['train'].select(range(split)).to_pandas()
output_list = []
input_list = []

#model = LLM("selfrag/selfrag_llama2_7b", download_dir="/gscratch/h2lab/akari/model_cache", dtype="half")
#sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)

def format_prompt(input, paragraph=None):
  prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)
  if paragraph is not None:
    prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
  return prompt
  
#preds = model.generate([format_prompt(query) for query in queries], sampling_params)
#for pred in preds:
#  print("Model prediction: {0}".format(pred.outputs[0].text))
  
if __name__ == "__main__":
  

  with torch.no_grad():
    for i in range(5):
        question = ds['train']['Question'][i]
        prompt = ds['train'][row]['Question'] + 'A)' + ds['train'][row]['Correct Answer'].replace('\n','') + ' B)' + ds['train'][row]['Incorrect Answer 1'].replace('\n','') + ' C)' + ds['train'][row]['Incorrect Answer 2'].replace('\n','') + ' D)' + ds['train'][row]['Incorrect Answer 3'].replace('\n','')
                
        input_list.append(question)
        retrieved_document = retriever.search_document_demo(question, 1)
        prompt = format_promt(prompt, paragraph = retrieved_document[0]['text'])
        response = llm_proposal(model,tokenizer, prompt)
        
        print(response)
        final_input = "Question: " + prompt + "\nAnswer: " + response[0] + '\n\n' + 'Extract only the final answer, without restating the question or explanation. Provide the answer as a short, direct response.'
        only_answer = llm_proposal(model,tokenizer,final_input)[0]
        output_list.append(only_answer)
    torch.cuda.empty_cache()
    # evaluator.evaluate(only_answer, ds['train']['Correct Answer'][i])
    run_evaluation(df,input_list,output_list, output_dir='/abr/coss39/rag_scale/output/5')


#retrieved_documents = retriever.search_document_demo(query_3, 5)
#prompts = [format_prompt(query_3, doc["title"] +"\n"+ doc["text"]) for doc in retrieved_documents]
#preds = model.generate(prompts, sampling_params)
#top_doc = retriever.search_document_demo(query_3, 1)[0]
  