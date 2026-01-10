from pathlib import Path



def get_config():
    return {
        "batch_size": 16,
        "num_epoch": 25,
        "lr": 10 ** -4,
        "seq_len": 256,
        "d_model": 256,
        "N": 6,
        "h": 8,
        "d_ff": 1024,
        "dropout": 0.1,
        "lang_src": "en",
        "lang_tar": "hi",
        "model_folder": "weights",
        "model_basename": "t_model_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch:str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)
