import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tar, src_lang, tar_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tar = tokenizer_tar
        self.src_lang = src_lang
        self.tar_lang = tar_lang
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor([tokenizer_tar.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tar.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tar.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tar_text = src_target_pair["translation"][self.tar_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tar.encode(tar_text).ids

        enc_num_padding = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding = self.seq_len - len(dec_input_tokens) - 1

        if (enc_num_padding < 0) or (dec_num_padding < 0):
            raise ValueError("Sentence too long")
        
        # SOS AND EOS added to source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding, dtype=torch.int64)
            ]
        )
        # Adding SOS
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding, dtype=torch.int64)
            ]
        )

        # Adding EOS
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return{
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)).int(),
            "label": label,
            "src_text": src_text,
            "tar_text": tar_text
        }


def causal_mask(size):
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0)
    return mask.int()

