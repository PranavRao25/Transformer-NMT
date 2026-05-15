"""
LEARNED POSITIONAL EMBEDDINGS
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm.auto import tqdm
import wandb
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu

from src.model import Transformer, make_src_mask, make_tgt_mask
# from model import Transformer, make_src_mask, make_tgt_mask
from src.lr_scheduler import NoamScheduler
from torch.nn.utils.rnn import pad_sequence
from src.dataset import Multi30kDataset

parent = Path(__file__).resolve().parent.parent

# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx, 
            label_smoothing=smoothing
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        return self.criterion(logits, target)

# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.
    """

    mode = "Train" if is_train else "Eval"
    
    try:
        total_batches = len(data_iter)
    except TypeError:
        total_batches = None
    
    loop = tqdm(data_iter, desc=f"Epoch {epoch_num} [{mode}]", leave=True, total=total_batches, mininterval=1.0)

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    scaler = torch.amp.GradScaler(enabled=(device == 'cuda')) # type: ignore # Mixed precision scaler

    for batch_idx, (src, tgt) in enumerate(loop):
        src, tgt = src.to(device), tgt.to(device)

        # Autoregressive shifting
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:].contiguous()

        pad_idx = loss_fn.pad_idx if hasattr(loss_fn, 'pad_idx') else 1
        src_mask = make_src_mask(src, pad_idx=pad_idx).to(device) # type: ignore
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=pad_idx).to(device) # type: ignore

        with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', enabled=(device == 'cuda')):
            logits = model(src, tgt_input, src_mask, tgt_mask)
            # Flatten to shape: [batch * seq_len, vocab_size] and [batch * seq_len]
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_labels.view(-1))

        if is_train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler is not None:
                scheduler.step()
        
        if batch_idx % 50 == 0:
            current_loss = loss.item()
            loop.set_postfix(loss=current_loss)

        total_loss += loss.item()

    return total_loss / len(data_iter)

# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.
    """
    model.eval()
    
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    
    for i in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
        
        with torch.no_grad():
            out = model.decode(memory, src_mask, ys, tgt_mask)
            
        prob = out[:, -1, :]
        _, next_word = torch.max(prob, dim=1)
        next_word_idx = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).fill_(next_word_idx).type(torch.long).to(device)], dim=1)
        if next_word_idx == end_symbol:
            break
            
    return ys

# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.
    """
    model.eval()
    
    start_symbol = tgt_vocab.token_to_id("[SOS]")
    end_symbol = tgt_vocab.token_to_id("[EOS]")
    pad_symbol = tgt_vocab.token_to_id("[PAD]")
    
    references = []
    candidates = []

    for src, tgt in tqdm(test_dataloader, desc="Calculating BLEU"):
        src = src.to(device)
        
        for i in range(src.size(0)):
            src_i = src[i:i+1]
            src_mask_i = make_src_mask(src_i, pad_idx=pad_symbol).to(device)
            
            pred_tokens = greedy_decode(
                model, src_i, src_mask_i, max_len, start_symbol, end_symbol, device
            )
            pred_indices = pred_tokens.squeeze(0).tolist()
            pred_words = [
                tgt_vocab.id_to_token(idx) for idx in pred_indices 
                if idx not in [start_symbol, end_symbol, pad_symbol]
            ]
            candidates.append(pred_words)
            
            target_indices = tgt[i].tolist()
            target_words = [
                tgt_vocab.id_to_token(idx) for idx in target_indices 
                if idx not in [start_symbol, end_symbol, pad_symbol]
            ]
            references.append([target_words])

    bleu_score = float(corpus_bleu(references, candidates) * 100) # type: ignore
    return bleu_score

# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """Save model + optimiser + scheduler state to disk."""
    
    model_config = {
        'src_vocab_size': model.src_embed.num_embeddings,
        'tgt_vocab_size': model.tgt_embed.num_embeddings,
        'd_model': model.d_model,
        'N': len(model.encoder.layers),
        'num_heads': model.encoder.layers[0].self_attn.num_heads, # type: ignore
        'd_ff': model.encoder.layers[0].ffn.linear1.out_features, # type: ignore
        'dropout': model.encoder.layers[0].dropout.p # type: ignore
    }

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'model_config': model_config
    }
    
    torch.save(state, path)

def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """Restore model (and optionally optimizer/scheduler) state from disk."""
    
    state = torch.load(path, map_location="cpu", weights_only=False)
    
    model.load_state_dict(state['model_state_dict'])
    
    if optimizer is not None and state.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])
        
    if scheduler is not None and state.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(state['scheduler_state_dict'])
        
    return state['epoch']

# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def collate_fn(batch, pad_idx):
    """
    Dynamically pads sequences in a batch to the maximum length within that batch.
    """
    src_list, tgt_list = [], []
    for src, tgt in batch:
        src_list.append(src)
        tgt_list.append(tgt)
    
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    
    return src_padded, tgt_padded

def load_data(config, device):
    train_dataset = Multi30kDataset(split='train')
    val_dataset = Multi30kDataset(split='validation')
    test_dataset = Multi30kDataset(split='test')

    src_vocab_size = len(train_dataset.src_vocab)
    tgt_vocab_size = len(train_dataset.tgt_vocab)
    print(f"Src vocab: {src_vocab_size}, Tgt vocab: {tgt_vocab_size}")
    
    pad_idx = train_dataset.PAD_IDX

    train_loader = DataLoader(
        train_dataset, 
        batch_size=int(config["batch_size"]),
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, pad_idx),
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=int(config["batch_size"]), 
        collate_fn=lambda b: collate_fn(b, pad_idx)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, # Single-sentence decoding for BLEU
        collate_fn=lambda b: collate_fn(b, pad_idx)
    )
    
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, src_vocab_size, tgt_vocab_size, pad_idx

def run_training_experiment() -> None:
    config = {
        "num_epochs": 20,
        "batch_size": 64,
        "d_model": 512,
        "n_layers": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "smoothing": 0.1,
        "warmup_steps": 4000,
        "learning_rate": 3e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "resume_from": "None"
    }
    
    model_name = f"Transformer_noam_labelsmoothing>0_nn_emb"

    wandb.init(project="da6401-a3", config=config)
    device = str(config["device"])

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, src_vocab_size, tgt_vocab_size, pad_idx = load_data(config, device)

    model = Transformer(int(config["d_model"]), int(config["n_layers"]),
                        int(config["num_heads"]), d_ff=int(config["d_ff"]), dropout=float(config["dropout"])).to(device)

    # model.decoder.output_projection.weight = model.target_embedding.embedding.weight
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
    )

    scheduler = NoamScheduler(optimizer, d_model=int(config["d_model"]),
                              warmup_steps=int(config["warmup_steps"]))

    start_epoch = 0
    if config["resume_from"] != "None":
        print(f"Resuming training from checkpoint: {config['resume_from']}")
        
        saved_epoch = load_checkpoint(
            str(parent / "model" / str(config["resume_from"])),
            model, 
            optimizer, 
            scheduler
        )
        start_epoch = saved_epoch + 1
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Resumed at Epoch {start_epoch} | Current LR: {current_lr:.2e}")

    loss_fn = LabelSmoothingLoss(vocab_size=tgt_vocab_size, pad_idx=pad_idx, smoothing=float(config["smoothing"]))

    wandb.watch(model, log="all", log_freq=50)
    
    best_val_loss = float('inf')
    for epoch in range(start_epoch, int(config["num_epochs"])):
        print(f"\n--- Epoch {epoch+1} ---")
        
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler, 
            epoch_num=epoch, is_train=True, device=device
        )
        
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None, 
            epoch_num=epoch, is_train=False, device=device
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]['lr']
        })

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, str(parent / "model" / f"{model_name}.pt"))

    print("\nStarting final evaluation...")
    load_checkpoint(str(parent / "model" / f"{model_name}.pt"), model)
    test_bleu = evaluate_bleu(model, test_loader, train_dataset.tgt_vocab, device=device)
    
    wandb.log({'test_bleu': test_bleu})
    print(f"Final Test BLEU Score: {test_bleu:.2f}")
    
    wandb.finish()

if __name__ == "__main__":
    run_training_experiment()
