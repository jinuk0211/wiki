
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from generator import LLM, llm_proposal, Generator
from cfg import cfg
from evaluator import GPQAEvaluator
#def llm_proposal(model=None,tokenizer=None,messages=None,model_name='qwen'):
#def LLM(model='qwen2.5'):
evaluator = GPQAEvaluator()
if __name__ == "__main__":
    print('initialize model')
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model,tokenizer=LLM(model='qwen2.5')
    messages = ['what is your name','who is the president of korean']
    question = 'who is the president of korean'
    generator = Generator(cfg, tokenizer, model, evaluator)
    #response = llm_proposal(model,tokenizer,messages,n=3)
    #print(response)
    print(generator.subanswer('',messages))
    print(generator.generate_subquestions(question,''))
    
    ori_q = 'When building a RAG system, which retrieval algorithm should I choose between FAISS and BM25?'
    subq = ['What are the advantages of using FAISS?','What are the advantages of using BM25?']
    suba = ['FAISS excels in large-scale vector search by providing fast approximate nearest-neighbor queries even over billions of vectors, leveraging GPU acceleration and efficient indexing.','BM25 is simple to implement and optimized for text-based retrieval, offering stable performance with minimal resources.']
    potential_score_output, final_answer =generator.final_output(ori_q,subq,suba)
    print(f'final_answer:{final_answer}')
