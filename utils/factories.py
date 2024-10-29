from typing import Tuple

from torch.utils.data import DataLoader, Dataset

from dataset.captioning_dataset import Batcher
from dataset.captioning_dataset import (
    # CaptioningDatasetWithFeatures as CaptioningDataset,
    # CaptioningDatasetFromHfpy as CaptioningDataset,
    DuplicatedCaptioningDatasetFromHfpy as CaptioningDataset,
)
from dataset.captioning_dataset import Vocab
from models.captioning_model import CaptioningModel
from models.transformer import (
    MemoryAugmentedEncoder,
    MeshedDecoder,
    ScaledDotProductAttentionMemory,
    Transformer,
)


def get_training_data(
    args,
) -> Tuple[Dataset, Dataset, Dataset]:
    train_data = CaptioningDataset(
        args.dataset_feat_path,
        args.dataset_ann_path,
        dataset_name=args.dataset,
        freq_threshold=5,
        split="train",
        feature_limit=args.feature_limit,
    )
    val_data = CaptioningDataset(
        args.dataset_feat_path,
        args.dataset_ann_path,
        dataset_name=args.dataset,
        freq_threshold=5,
        split="val",
        feature_limit=args.feature_limit,
    )
    test_data = CaptioningDataset(
        args.dataset_feat_path,
        args.dataset_ann_path,
        dataset_name=args.dataset,
        freq_threshold=5,
        split="test",
        feature_limit=args.feature_limit,
    )
    return train_data, val_data, test_data


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        collate_fn=Batcher(dataset.vocab),
    )


def get_model(args: any, vocab: Vocab) -> CaptioningModel:
    encoder = MemoryAugmentedEncoder(
        args.n,
        vocab.stoi["<pad>"],
        attention_module=ScaledDotProductAttentionMemory,
        attention_module_kwargs={"m": args.m},
        d_in=args.meshed_emb_size,
        dropout=args.dropout,
    )
    decoder = MeshedDecoder(
        len(vocab), 54, args.n, vocab.stoi["<pad>"], dropout=args.dropout
    )
    model = Transformer(vocab.stoi["<bos>"], encoder, decoder)

    param = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {trainable}/{param} parameters")
    return model
