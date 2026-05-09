import spacy
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
from collections import Counter

class Vocabulary:
    """Native Vocabulary manager to replace deprecated torchtext."""
    def __init__(self, min_freq=1, specials=None):
        self.min_freq = min_freq
        self.specials = specials or []
        self.stoi = {}
        self.itos = {}
        self.default_idx = None
        
    def build_from_iterator(self, iterator):
        counter = Counter()
        for tokens in iterator:
            counter.update(tokens)
            
        for idx, token in enumerate(self.specials):
            self.stoi[token] = idx
            self.itos[idx] = token
            
        idx = len(self.specials)
        for token, freq in counter.most_common():
            if freq >= self.min_freq and token not in self.stoi:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1
                
    def set_default_index(self, idx):
        self.default_idx = idx
        
    def __getitem__(self, token):
        """Allows src_vocab[token] syntax during tensorization."""
        return self.stoi.get(token, self.default_idx)
        
    def token_to_id(self, token):
        """Satisfies train.py greedy_decode / evaluate_bleu contract."""
        return self.__getitem__(token)
        
    def id_to_token(self, idx):
        """Satisfies train.py greedy_decode / evaluate_bleu contract."""
        return self.itos.get(idx, self.specials[0] if self.specials else "[UNK]")
        
    def __len__(self):
        return len(self.stoi)

class Multi30kDataset(Dataset):
    def __init__(self, split='train', src_lang='de', tgt_lang='en', min_freq=2):
        """
        Loads the Multi30k dataset and prepares tokenizers and vocabularies.
        Inherits from torch.utils.data.Dataset for seamless DataLoader integration.
        """
        self.split = split
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # 1. Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        print(f"Loading '{split}' split of multi30k...")
        self.dataset = load_dataset("bentrevett/multi30k", split=split)

        try:
            self.spacy_src = spacy.load('de_core_news_sm')
            self.spacy_tgt = spacy.load('en_core_web_sm')
        except OSError:
            raise OSError("SpaCy models not found. Please download them using the python -m spacy download commands.")

        self.UNK_IDX, self.PAD_IDX, self.SOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        self.special_tokens = ['[UNK]', '[PAD]', '[SOS]', '[EOS]']

        print("Building vocabularies...")
        self.src_vocab, self.tgt_vocab = self.build_vocab(min_freq=min_freq)

        print("Tokenizing and tensorizing data...")
        self.processed_data = self.process_data()

    def tokenize_src(self, text):
        """Yields string tokens from the source text."""
        return [tok.text for tok in self.spacy_src.tokenizer(text)]

    def tokenize_tgt(self, text):
        """Yields string tokens from the target text."""
        return [tok.text for tok in self.spacy_tgt.tokenizer(text)]

    def build_vocab(self, min_freq):
        """
        Builds the vocabulary mapping natively.
        """
        train_data = load_dataset("bentrevett/multi30k", split='train')

        def yield_src_tokens():
            for item in train_data:
                yield self.tokenize_src(item[self.src_lang]) # type: ignore

        def yield_tgt_tokens():
            for item in train_data:
                yield self.tokenize_tgt(item[self.tgt_lang]) # type: ignore

        src_vocab = Vocabulary(min_freq=min_freq, specials=self.special_tokens)
        src_vocab.build_from_iterator(yield_src_tokens())
        src_vocab.set_default_index(self.UNK_IDX)

        tgt_vocab = Vocabulary(min_freq=min_freq, specials=self.special_tokens)
        tgt_vocab.build_from_iterator(yield_tgt_tokens())
        tgt_vocab.set_default_index(self.UNK_IDX)

        return src_vocab, tgt_vocab

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary, wrapped in PyTorch tensors.
        """
        processed = []
        for item in tqdm(self.dataset, desc=f"Processing {self.split} Data"):
            # 1. Tokenize into strings
            src_tokens = self.tokenize_src(item[self.src_lang]) # type: ignore
            tgt_tokens = self.tokenize_tgt(item[self.tgt_lang]) # type: ignore

            # 2. Convert to indices and append SOS/EOS tokens
            src_indices = [self.SOS_IDX] + [self.src_vocab[token] for token in src_tokens] + [self.EOS_IDX]
            tgt_indices = [self.SOS_IDX] + [self.tgt_vocab[token] for token in tgt_tokens] + [self.EOS_IDX]

            # 3. Store as PyTorch tensors
            processed.append({
                'src': torch.tensor(src_indices, dtype=torch.long),
                'tgt': torch.tensor(tgt_indices, dtype=torch.long)
            })
            
        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]['src'], self.processed_data[idx]['tgt']
