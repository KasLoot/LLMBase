from huggingface_hub import hf_hub_download

REPO_ID = "meta-math/MetaMathQA"
FILE_NAME = "MetaMathQA-395K.json"

hf_hub_download(repo_id=REPO_ID, repo_type="dataset",filename=FILE_NAME, local_dir="/Users/liuyuxin/Documents/datasets/")
