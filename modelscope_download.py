from modelscope import snapshot_download
model_dir = snapshot_download(model_id="tiansz/bert-base-chinese", cache_dir="./model")
