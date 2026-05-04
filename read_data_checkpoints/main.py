import os
import glob
import math
import torch
import numpy as np
from torch.utils.data import DataLoader, ChainDataset
from tfrecord.torch.dataset import TFRecordDataset
from sentencepiece import sentencepiece_model_pb2
import sentencepiece as sentencepiece_processor
from tqdm import tqdm
import litdata as ld

def pad_collate(batch, pad_value=0):
    """
    Collate function for zero-padding 1D sequences.
    Assumes each item is a dict with keys like 'input_ids', 'label', etc.
    """
    out = {}
    keys = batch[0].keys()

    for key in keys:
        values = [x[key] for x in batch]

        if isinstance(values[0], (list, np.ndarray, torch.Tensor)) and len(values[0]) != 0:
            lengths = [len(v) for v in values]
            max_len = max(lengths)
            padded = []
            for v in values:
                v_tensor = torch.tensor(v, dtype=torch.long)
                if v_tensor.size(0) < max_len:
                    pad = torch.full((max_len - v_tensor.size(0),), pad_value, dtype=torch.long)
                    v_tensor = torch.cat([v_tensor, pad], dim=0)
                padded.append(v_tensor)
            out[key] = torch.stack(padded)
        else:
            # For scalars (e.g., labels)
            out[key] = torch.tensor(values)
    
    return out
 
def create_tfrecord_dataloader(
    tfrecord_dir,
    description,
    batch_size,
    split,
    num_workers=1
):
    tfrecord_files = sorted(glob.glob(os.path.join(tfrecord_dir, f'{split}.tfrecord-*')))
    
    datasets = []
    for tf_file in tfrecord_files:
        ds = TFRecordDataset(
            data_path=tf_file,
            index_path=None,
            description=description,
            shuffle_queue_size=0,
        )
        datasets.append(ds)
    
    combined_dataset = ChainDataset(datasets)

    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=pad_collate if batch_size > 1 else None,
    )

    return dataloader

if __name__ == "__main__":
    tfrecord_dir = os.environ.get("OPTFORMER_BBOB_TRAIN_DIR", "data/cached_bbob_train/")
    tfrecord_dir_eval = os.environ.get("OPTFORMER_BBOB_EVAL_DIR", "data/cached_bbob_eval/")

    # copy from https://github.com/google-research/optformer/blob/neurips22/optformer/t5x/vocabularies.py
    with open("sentencepiece.model", "rb") as f:
        max_integer_tokens = 1000
        sp_model = sentencepiece_model_pb2.ModelProto.FromString(f.read())
        extra_tokens = ['<' + str(n) + '>' for n in range(max_integer_tokens)]
        piece2idx = {piece.piece: i for i, piece in enumerate(sp_model.pieces)}
        
        for extra_token in extra_tokens:
            # Add exponential length score for longest match.
            extra_token = f"{extra_token}"
            score = math.exp(len(extra_token))
            if extra_token in piece2idx:
                # print(
                #     "Overwriting piece: %s, score: %s -> %s",
                #     extra_token,
                #     sp_model.pieces[piece2idx[extra_token]].score,
                #     score,
                # )
                sp_model.pieces[piece2idx[extra_token]].score = score
            else:
                # print("Adding piece: %s, score: %s", extra_token, score)
                sp_model.pieces.add(
                    piece=extra_token,
                    score=score,
                    type=sentencepiece_model_pb2.ModelProto.SentencePiece.USER_DEFINED,
                )
        
        tokenizer = sentencepiece_processor.SentencePieceProcessor()
        tokenizer.LoadFromSerializedProto(sp_model.SerializeToString())

    # Define your feature schema
    feature_description = {
        "inputs": "int",  
        "targets": "int",
        "inputs_pretokenized": "byte",
        "targets_pretokenized": "byte",
    }

    # Reset dataloader
    dataloader = create_tfrecord_dataloader(
        tfrecord_dir=tfrecord_dir,
        description=feature_description,
        batch_size=1,
        split="train",
        num_workers=8
    )
    
    # Print a sample batch to check sequence lengths
    for batch in dataloader:
        inputs = batch["inputs"]
        targets = batch["targets"]
        print(f"Sample input shape: {inputs.shape}, Sample target shape: {targets.shape}")
        print(f"Context: {batch['inputs_pretokenized'][0].decode('utf-8')}")
        print(f"Optimization trace: {batch['targets_pretokenized'][0].decode('utf-8')}")

        assert tokenizer.decode(inputs[0].tolist()) == batch["inputs_pretokenized"][0].decode("utf-8")
        assert tokenizer.decode(targets[0].tolist()) == batch["targets_pretokenized"][0].decode("utf-8")
        break
