# Readme

### Download and run Ollama without root access
```shell
curl -L https://ollama.com/download/ollama-linux-amd64 -o ./ollama
sudo chmod +x /usr/bin/ollama
/usr/bin/ollama serve
/usr/bin/ollama run llama3
```

### Switch to GPU partition
```shell
 sinfo
 sinfo --Node  --partition=gpu
 sinfo --partition=gpu --format="%P %t %D %C %G"
 sinfo --partition=gpu --format="%N %P %t %D %C %G"
 squeue
 srun --partition=gpu --gres=gpu:1 --pty $SHELL
 srun --partition=gpu --gres=gpu:v100_32:1 --pty $SHELL
 srun --partition=cpu --pty $SHELL
```

### CRAG benchmark
```shell
git clone https://github.com/facebookresearch/CRAG.git
cd CRAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install auto-gptq --no-build-isolation
# pip install auto-gptq --no-build-isolation --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
pip install "vllm==0.4.2"
pip install "debugpy==1.8.0"

# bzcat  crag_task_3_dev_v4.tar.bz2.part* | bzip2 >crag_task_3_dev_v4_merged.tar.bz2
cat data/crag_task_3_dev_v4.tar.bz2.part* > data/crag_task_3_dev_v4.tar.bz2

pip install huggingface_hub[hf_transfer]
huggingface-cli login
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir-use-symlinks False \
    --local-dir models/meta-llama/Meta-Llama-3-8B-Instruct \
    --exclude *.pth # These are alternates to the safetensors hence not needed

# for local testing only
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    meta-llama/Llama-3.2-3B-Instruct \
    --local-dir-use-symlinks False \
    --local-dir models/meta-llama/Llama-3.2-3B-Instruct \
    --exclude *.pth # These are alternates to the safetensors hence not needed
    
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
   sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir-use-symlinks False \
    --local-dir models/sentence-transformers/all-MiniLM-L6-v2 \
    --exclude *.bin *.h5 *.ot # These are alternates to the safetensors hence not needed
    
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download   intfloat/e5-base-v2   --local-dir-use-symlinks False     --local-dir models/intfloat/e5-base-v2     --exclude *.pth

# python local_evaluation.py
python local_evaluation.py --dataset-path data/crag_task_1_and_2_dev_v4.jsonl.bz2 --generate-only


cd mock_api
# python3 -m venv .venv
# source .venv/bin/activate
pip install -r requirements.txt
pip install "numpy<2.0.0"
uvicorn server:app --reload


```

### FlashRAG benchmark
```shell
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download  intfloat/e5-large-v2   --local-dir-use-symlinks False     --local-dir models/intfloat/e5-large-v2     --exclude *.pth

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download     meta-llama/Meta-Llama-3-8B-Instruct     --local-dir-use-symlinks False     --local-dir models/meta-llama/Meta-Llama-3-8B-Instruct     --exclude *.pth # These are alternates to the safetensors hence not needed

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download     meta-llama/Llama-3.2-3B-Instruct     --local-dir-use-symlinks False     --local-dir models/meta-llama/Llama-3.2-3B-Instruct     --exclude *.pth

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download  selfrag/selfrag_llama2_7b     --local-dir-use-symlinks False     --local-dir models/selfrag/selfrag_llama2_7b     --exclude *.pth

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download RUC-NLPIR/FlashRAG_datasets   --local-dir-use-symlinks False     --local-dir datasets/FlashRAG_datasets     --exclude *.pth --repo-type dataset

# adaptive rag
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download illuminoplanet/combined_flan_t5_xl_classifier     --local-dir-use-symlinks False     --local-dir models/illuminoplanet/adaptive_rag_unnoficial_t5_classifier     --exclude *.pth

wget https://archive.org/download/enwiki-20181220/enwiki-20181220-pages-articles.xml.bz2
python preprocess_wiki.py --dump_path "/home/kucerj56/datasets/corpus/enwiki-20181220-pages-articles.xml.bz2" --save_path "/home/kucerj56/datasets/corpus/enwiki-20181220.json" --chunk_by recursive --tokenizer_name_or_path "/home/kucerj56/models/intfloat/e5-large-v2" --chunk_size 1024 --num_workers 18

python -m flashrag.retriever.index_builder   --retrieval_method e5   --model_path ~/models/intfloat/e5-base-v2/   --corpus_path ~/datasets/FlashRAG_datasets/retrieval-corpus/domainrag_text_corpus.jsonl   --save_dir ~/indexes/   --use_fp16   --max_length 512   --batch_size 256   --pooling_method mean   --sentence_transformer   --faiss_type Flat
# TODO add diff index type
python run_exp.py --method_name 'naive'                   --split 'test'                   --dataset_name 'nq'                   --gpu_id '0'



python scripts/chunk_doc_corpus.py --input_path "/Users/jakubkucera-sch/Documents/diplomka/llm-rag-dip/nss.jsonl" --output_path "nss_chuncked.json"


python run_exp_split.py --method_name 'zero-shot'                   --split 'test'                   --dataset_name 'nq'                   --gpu_id '0' --generate-only
python run_eval_only.py --config-file "/Users/jakubkucera-sch/Documents/diplomka/FlashRAG/examples/methods/output/nq_2025_03_02_19_25_zero-shot/config.yaml" --generated-dataset-path "/Users/jakubkucera-sch/Documents/diplomka/FlashRAG/examples/methods/output/nq_2025_03_02_19_25_zero-shot/generated.json"   
```

```shell
python scripts/chunk_doc_corpus.py --input_path ~/datasets/crag_data/flashrag_crag_unchunked_corpus.jsonl --output_path ~/datasets/crag_data/flashrag_crag_unchunked_corpus_chunked.jsonl --chunk_by recursive --chunk_size 512

python -m flashrag.retriever.index_builder   --retrieval_method e5   --model_path ~/models/intfloat/e5-base-v2/   --corpus_path /home/kucerj56/datasets/crag_data/flashrag_crag_chunked_corpus.jsonl   --save_dir ~/datasets/index   --use_fp16   --max_length 512   --batch_size 256   --pooling_method mean   --sentence_transformer   --faiss_type Flat
python -m flashrag.retriever.index_builder   --retrieval_method e5   --model_path ~/models/intfloat/e5-base-v2/   --corpus_path /home/kucerj56/datasets/crag_data/flashrag_crag_unchunked_corpus_chunked.jsonl   --save_dir ~/indexes/   --use_fp16   --max_length 512   --batch_size 256   --pooling_method mean   --sentence_transformer   --faiss_type Flat

# python -m spacy download cs_core_news_sm


```

waivate
```shell
git clone git@github.com:weaviate/weaviate.git
cd weaviate
git checkout d45e4fd
# wget https://go.dev/dl/go1.24.1.linux-amd64.tar.gz
go mod download
go build -o weaviate-server ./cmd/weaviate-server
./weaviate-server --host 0.0.0.0 --port 8080 --scheme HTTP
```

```shell
 ssh -NfL localhost:9999:localhost:9999  kucerj56@cluster.in.fit.cvut.cz
# ssh -NfL localhost:8888:localhost:8888  kucerj56@cluster.in.fit.cvut.cz
 ssh -NfL localhost:9999:localhost:9999 kucerj56@cf-prod-node-031
# ssh -NfL localhost:8888:localhost:8888 kucerj56@cf-prod-node-11
```

```shell
 ssh -N -f -R 5001:localhost:5001 kucerj56@cluster.in.fit.cvut.cz
 ssh -N -f -R 5001:localhost:5001 kucerj56@cf-prod-node-011

```


```shell
#!/bin/sh
cd /home/kucerj56/FlashRAG/experiments/test_sh_script
/home/kucerj56/FlashRAG/.venv/bin/python /home/kucerj56/FlashRAG/examples/methods/run_exp.py --method_name 'zero-shot'                   --split 'test'                   --dataset_>
```

```shell
sbatch --partition=cpu  /home/kucerj56/FlashRAG/experiments/test_sh_script/script.sh

#        --exclude=cf-prod-node-021,cf-prod-node-022 \
sbatch --partition=gpu \
       --gres=gpu:1 \
        --exclude=cf-prod-node-021,cf-prod-node-022,cf-prod-node-024 \
       script.sh

```
