
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from generator import LLM, llm_proposal, Generator
from cfg import cfg
from prompt import rag_prompt
from evaluator import GPQAEvaluator
from verify import initialize_value_model, probability_subanswer_question, probability_subquestion_question
import numpy as np
#def llm_proposal(model=None,tokenizer=None,messages=None,model_name='qwen'):
#def LLM(model='qwen2.5'):
evaluator = GPQAEvaluator()
if __name__ == "__main__":
    print('initialize model')
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    global global_value_model, global_tokenizer
    model,tokenizer=LLM(model='qwen2.5')
    global_value_model = model 
    gloval_tokenizer = tokenizer
    messages = ['what is your name','who is the president of korean']
    question = 'who is the president of korean'
    subquestions =  ['what is your name','who is the president of korean','what is the problem']
    value_list = []
    for i in subquestions:
      value = probability_subquestion_question(question, i) # probablistic score
      value_list.append(value)
    top_indices = np.argsort(value_list)[::-1][:3]
    top3_subquestions = [subquestions[i] for i in top_indices] #top3 subquestions
    last_indices = np.argsort(value_list)[::-1][3:]
    bad_subquestions = [subquestions[i] for i in last_indices]  
    print(f'value:{value_list}')     
    
    generator = Generator(cfg, tokenizer, model, evaluator)
    #response = llm_proposal(model,tokenizer,messages,n=3)
    #print(response)
    #print(generator.subanswer('',messages))
    #subquestions = generator.generate_subquestions(question,'')
    #print(subquestions)
    
    ori_q = 'When building a RAG system, which retrieval algorithm should I choose between FAISS and BM25?'
    context = 'What are the advantages of using FAISS? FAISS excels in large-scale vector search by providing fast approximate nearest-neighbor queries even over billions of vectors, leveraging GPU acceleration and efficient indexing'
    subq = ['What are the advantages of using FAISS?','What are the advantages of using BM25?']
    suba = ['FAISS excels in large-scale vector search by providing fast approximate nearest-neighbor queries even over billions of vectors, leveraging GPU acceleration and efficient indexing.','BM25 is simple to implement and optimized for text-based retrieval, offering stable performance with minimal resources.']
    potential_score_output, final_answer =generator.final_output(ori_q,subq,suba)
    score = generator.score(rag_prompt.format(context=context,question=ori_q))
    print(f'score:{score}')
    print(f'final_answer:{final_answer}')
