import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from tqdm import tqdm
import warnings


def greedy_decode(model, source, source_mask,tokenizer_src, tokenizer_tar, max_len, device):
    sos_idx = tokenizer_tar.token_to_id("[SOS]")
    eos_idx = tokenizer_tar.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input witht the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Next token
        prob = model.project(out[:,-1])

        # Token with the highest probability
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tar, max_len, device, print_msg, global_state,  writer, num_examples=2):
    model.eval()
    count = 0 
    

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tar, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tar_text'][0]
            model_out_text = tokenizer_tar.decode(model_out.detach().cpu().numpy())


            print_msg(""*console_width)
            print_msg(f"SOURCE: {source_text}")
            print_msg(f"TARGET: {target_text}")
            print_msg(f"PREDICTED: {model_out_text}")

            if count == num_examples:
                break
        



def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]



def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_ds(config):

    ds_raw = load_dataset(
    "opus100",
    f'{config["lang_src"]}-{config["lang_tar"]}',
    split="train"
)



    tokenizer_source = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_target = get_or_build_tokenizer(config, ds_raw, config["lang_tar"])

    train_ds_size = int(0.9* len(ds_raw)) # 90% for Training
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_source, tokenizer_target, config["lang_src"], config["lang_tar"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_source, tokenizer_target, config["lang_src"], config["lang_tar"], config["seq_len"])

    max_len_src = 0
    max_len_tar = 0

    for item in ds_raw:
        src_ids = tokenizer_source.encode(item["translation"][config["lang_src"]]).ids
        tar_ids = tokenizer_target.encode(item["translation"][config["lang_tar"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tar = max(max_len_tar, len(tar_ids))
    

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tar}")

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False
    )

    return train_dataloader, val_dataloader, tokenizer_source, tokenizer_target

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'], config['N'], config['h'], config['dropout'], config['d_ff'])
    return model

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_source, tokenizer_target = get_ds(config)
        # Add this right after loading your tokenizers
    print(f"Source tokenizer vocab size: {tokenizer_source.get_vocab_size()}")
    print(f"Target tokenizer vocab size: {tokenizer_target.get_vocab_size()}")
    print(f"SOS token ID: {tokenizer_source.token_to_id('[SOS]')}")
    print(f"EOS token ID: {tokenizer_source.token_to_id('[EOS]')}")
    print(f"PAD token ID: {tokenizer_source.token_to_id('[PAD]')}")
    model = get_model(config, tokenizer_source.get_vocab_size(), tokenizer_target.get_vocab_size()).to(device)

    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_source.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config["num_epoch"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch{epoch:02d}")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            # Run tensors through Transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_ouput =  model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_ouput)
            
            label = batch["label"].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss:" : f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad()

            # Run Val
            run_validation(model, val_dataloader, tokenizer_source, tokenizer_target, config["seq_len"], device, lambda msg: batch_iterator.write(msg), global_step, writer)

            global_step += 1
        run_validation(model, val_dataloader, tokenizer_source, tokenizer_target, config["seq_len"], device, lambda msg: batch_iterator.write(msg), global_step, writer)
       
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename) 

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)

