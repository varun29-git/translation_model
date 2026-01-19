# English to Hindi Translation Model

This repository implements an encoder–decoder Transformer for English to Hindi translation using PyTorch, with all components written clearly and explicitly.
The model architecture follows the original Transformer architecture from the paper [“Attention Is All You Need”](https://arxiv.org/pdf/1706.03762).


## Model Architecture 
*`model.py`*
* Learned input embeddings scaled by √d_model for source and target.
* Sinusoidal positional embeddings added to token embeddings for order information.
* Multi-head attention with query, key, and value projections, head splitting, and recombination.
* Residual connections with pre-layer normalization.
* Feed-forward networks inside each block (linear -> ReLU -> linear -> dropout).
* Encoder stack of self-attention + feed-forward blocks.
* Decoder stack with masked self-attention, encoder–decoder attention, and feed-forward layers.
* Projection layer producing log-probabilities over the target vocabulary.


## Attention & Masking

* Padding masks prevent attention on padded tokens.
* Causal masks ensure autoregressive decoding in the decoder.


## Data & Tokenization
*`dataset.py`*
* Custom bilingual dataset class builds:
  * Encoder and decoder inputs.
  * `[SOS]`, `[EOS]`, `[PAD]` token handling.
  * Encoder and decoder attention masks
* Tokenizers are trained from scratch using a WordLevel model.
* Training uses a subset of OPUS-100 English–Hindi data.


## Training 
*`train.py`*
* Full training loop with:
  * Batching and shuffling
  * Cross-entropy loss with padding masking
  * Label smoothing
* Uses Adam optimizer and logs metrics via TensorBoard.
* Saves model checkpoints.
  

## Training Observations
*`English_to_Hindi_Transformer_Training.ipynb`*
* Training started with high cross-entropy loss (~6), reflecting random initialization.
* Loss steadily decreased over epochs, reaching ~2 by the final checkpoint.
* Despite decreasing loss, translations remain imperfect, highlighting the gap between token-level likelihood optimization and sentence-level translation quality.


## Inference Behavior & Observations
*`inference.ipynb`*
During inference, the model shows clearly different behavior depending on how close the input sentence is to the training distribution.

# 1. Common conversational sentences

Example:
*“How are you doing today?”
“Do you know who I am?”*

* These sentences appear frequently (or in very similar forms) in the training data.
* The model has seen enough examples to learn stable word order, verb usage, and question structure.
* As a result, translations are fluent and grammatically correct.

# 2. Sentences with proper nouns or rare constructions

Example:
*“My name is Varun”
“I am a writer”*

* Proper nouns like names are rare or inconsistently represented in the dataset.
* The tokenizer may split or map them to less meaningful sub-tokens.
* The model defaults to common sentence templates (“मेरा नाम है …”) but fails to complete them correctly.

# 3. Out-of-distribution inputs

Example:
*“Namaste”*

* This input is outside the English–Hindi translation distribution used during training.
* The model has no learned mapping for this token sequence.
* As a result, decoding collapses into near-random or meaningless outputs.


## Notes

This project was built as a learning exercise to understand how modern sequence-to-sequence models work end-to-end from tokenization and masking to attention, residual connections, and training dynamics.

Training behavior (loss trends, translation quality, and failure cases) reflects the expected limitations of a small Transformer trained from scratch on limited data with limited compute. Improving fluency and handling proper nouns would require larger datasets, longer training, and stronger decoding strategies.
