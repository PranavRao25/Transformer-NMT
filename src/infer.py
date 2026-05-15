import torch
import yaml
import json
import os
from pathlib import Path
import argparse
from src.model import Transformer
from src.dataset import Multi30kDataset

parent = Path(__file__).resolve().parent.parent

def load_vocab():
    train_dataset = Multi30kDataset(split='train')
    src_vocab_size = len(train_dataset.src_vocab)
    tgt_vocab_size = len(train_dataset.tgt_vocab)

    return src_vocab_size, tgt_vocab_size

def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description='Train a Transformer')
    
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=20,
        action="store",
        dest="epochs",
        help="specify the no of epochs"
    )
    
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=64,
        action="store",
        dest="batch_size",
        help="specify the batch size"
    )

    parser.add_argument(
        "-d", "--d_model",
        type = int,
        default = 512,
        action = "store",
        dest = "d_model",
        help = "specify the input dimension"
    )

    parser.add_argument(
        "-y", "--n_layers",
        type = int,
        default = 6,
        action = "store",
        dest = "n_layers",
        help = "specify the no of layers"
    )

    parser.add_argument(
        "-a", "--num_heads",
        type = int,
        default = 8,
        action = "store",
        dest = "num_heads",
        help = "specify the no of heads"
    )

    parser.add_argument(
        "-f", "--d_ff",
        type = int,
        default = 2048,
        action = "store",
        dest = "d_ff",
        help = "specify the dimension of ffn"
    )

    parser.add_argument(
        "-p", "--dropout",
        type = float,
        default = 0.1,
        action = "store",
        dest = "dropout",
        help = "specify the dropout"
    )

    parser.add_argument(
        "-s", "--smoothing",
        type = float,
        default = 0.1,
        action = "store",
        dest = "smoothing",
        help = "specify the smoothing"
    )

    parser.add_argument(
        "-u", "--warmup_steps",
        type = int,
        default = 4000,
        action = "store",
        dest = "warmup_steps",
        help = "specify the warmup steps"
    )

    parser.add_argument(
        "-l", "--learning_rate",
        type = int,
        default = 128,
        action = "store",
        dest = "learning_rate",
        help = "specify the learning rate"
    )

    parser.add_argument(
        "-v", "--device",
        type = str,
        default = "cuda",
        action = "store",
        dest = "device",
        help = "specify the device"
    )

    parser.add_argument(
        "-r", "--resume_from",
        type = str,
        action = "store",
        dest = "resume_from",
        help = "specify the resume checkpoint"
    )

    parser.add_argument(
        "-m", "--model_path",
        type = str,
        default = "https://drive.google.com/file/d/1iwWHsn4nRr32OaPIILzLkMtMFPVKolDG/view?usp=sharing",
        action = "store",
        dest = "model_path",
        help = "specify the model path"
    )

    parser.add_argument(
        "-w", "--wandb",
        type = str,
        default = "da6401-a3",
        action = "store",
        dest = "wandb",
        help = "specify the wandb project"
    )

    return parser.parse_args()

# config = {
#     "num_epochs": 20,
#     "batch_size": 64,
#     "d_model": 512,
#     "n_layers": 6,
#     "num_heads": 8,
#     "d_ff": 2048,
#     "dropout": 0.1,
#     "smoothing": 0.1,
#     "warmup_steps": 4000,
#     "learning_rate": 3e-4,
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "resume_from": "None"
# }

# model_paths = [
#     # "https://drive.google.com/file/d/1MQGn5JmRDy5zxy2ZRnbvvs95ARUZGoqx/view?usp=sharing",  # Transformer_noam_labelsmoothing>0_1
#     # "https://drive.google.com/file/d/1SepO12yUy86Pt8NE2S56kJfpKO26hMmh/view?usp=sharing", # OKAY Transformer_constant_lr_labelsmoothing>0_sin
#     # "https://drive.google.com/file/d/1iwWHsn4nRr32OaPIILzLkMtMFPVKolDG/view?usp=sharing", # OKAY (Use this) Transformer_noam_labelsmoothing>0_nn_emb
#     # "https://drive.google.com/file/d/1N5OOl-DiaijTIC9PYWkUdUI_7sKshd58/view?usp=sharing",  # BAD Transformer_noam_labelsmoothing>0_sin_unscaled
#     # "https://drive.google.com/file/d/1JFUqitbSvMGUw3LzngAxZ_RtlRQQIoVD/view?usp=sharing",  # OKAY Transformer_noam_labelsmoothing>0
#     # "https://drive.google.com/file/d/1YqWIi_K2P1JJT63wWGQ6gPJSc0251wSf/view?usp=sharing",  # OKAY Transformer_noam_labelsmoothing0_sin
# ]

if __name__ == "__main__":
    args = parse_arguments()
    config = vars(args)

    # model_name = f"Transformer_noam_labelsmoothing>0_sin_unscaled"
    device = str(config["device"])

    model = Transformer(int(config["d_model"]), int(config["n_layers"]),
                            int(config["num_heads"]), d_ff=int(config["d_ff"]), dropout=float(config["dropout"]),
                            model_path=str(config["model_path"])
                            ).to(device)

    src_sents = [
        "Guten Tag.",
        "Mein Name ist Mondly.",
        "Wie geht's?",
        "Kerstin ist 25 Jahre alt. Sie studiert Medizin in Frankfurt und hat viele Freunde, weil sie sympatisch und freundlich ist.",
        "Sie geht mit einem jungen Mann aus, der Robert heisst.",
        "Er ist 30 Jahre alt und arbeitet in Stuttgart: er ist ein Ingenieur.",
        "Robert geht gerne am Abend aus, geht ins Kino, und trifft sich mit seinen Freunden.",
        "Kerstin und Robert sind seit drei Jahren verlobt.",
        "Sie wohnen in verschiedenen Städten, aber zum Glück sind Frankfurt und Stuttgart nicht sehr weit entfernt."
        "Kerstin hat die Wohnungsschlüssel von Robert und Robert hat die Schlüssel von Kerstin: die beiden haben keine Geheimnisse.",
        "Heute ist Mittwoch. Kerstin hat keine Vorlesung in der Universität.",
        "Sie möchte Robert eine Überraschung bereiten, weil heute ein besonderer Tag ist. Für Kerstin ist es tatsächlich ein sehr wichtiger Tag."
    ]

    tgt_sents = [
        "Good Afternoon",
        "My Name is Mondly",
        "How are you?",
        "Kerstin is 25 years old. She is studying medicine at Frankfurt. She has lots of friends because she is likable and kind.",
        "She is going out with a young man called Robert.",
        "He is 30 years old and works at Stuttgart, as an engineer.",
        "Robert loves going out in the evenings, to the cinema or to meet friends.",
        "Kerstin and Robert have been engaged for three years.",
        "They live in different cities but luckily Frankfurt and Stuttgart are not very far apart.",
        "Kerstin has the keys to Robert’s house and Robert has those of Kerstin’s. The two young people don’t have any secrets.",
        "Today is Wednesday. Kerstin has no lessons at the university.",
        "She wants to give Robert a surprise because today is a special day. For Kerstin, in fact it is a very important day."
    ]

    print("Inferring")
    pred_sents = list(map(model.infer, src_sents))

    for i,j in zip(pred_sents, tgt_sents):
        print(i, j)
