pip install gdown
pip install huggingface_hub
pip install faiss-cpu
pip install vllm transformers
pip install easydict
pip install datasets
pip install fuzzywuzzy
pip install collections
git clone https://github.com/AkariAsai/self-rag.git -q
cd self-rag
pip install -r requirements.txt -q
cd retrieval_lm
mkdir -p enwiki_2020_intro_only/enwiki_dec_2020_contriever_intro
gdown https://drive.google.com/uc?id=1-24buVYsvSU4laZW9FXOQG8P6bMucro8 -O enwiki_2020_intro_only/enwiki_2020_dec_intro_only.jsonl
gdown https://drive.google.com/uc?id=1YasSXY4_mRaNkgkQEeRA6y-y0WH5diN8 -O enwiki_2020_intro_only/enwiki_dec_2020_contriever_intro/passages_01
git clone https://github.com/jinuk0211/rag_scale.git
huggingface-cli login
