import json
import h5py
import os
from collections import Counter
from itertools import chain
from typing import List

import json
import numpy as np
import torch
from torch.utils.data import Dataset


def preprocess_caption(caption: str) -> str:
    # Clean sentence list following: https://cs.stanford.edu/people/karpathy/cvpr2015.pdf Section 4
    caption = caption.lower()

    # Disgard non-alphanumeric characters
    non_alphanumeric = [chr(i) for i in range(33, 128) if not chr(i).isalnum()]

    for char in non_alphanumeric:
        caption = caption.replace(char, "")
    while "  " in caption:
        caption = caption.replace("  ", " ")

    return caption.strip()


def preprocess_captions(captions: List[str]) -> List[str]:
    # Clean sentence list following: https://cs.stanford.edu/people/karpathy/cvpr2015.pdf Section 4
    captions = [caption.lower() for caption in captions]

    # Disgard non-alphanumeric characters
    non_alphanumeric = [chr(i) for i in range(33, 128) if not chr(i).isalnum()]
    cleaned = []

    for sentence in captions:
        for char in non_alphanumeric:
            sentence = sentence.replace(char, "")
        while "  " in sentence:
            sentence = sentence.replace("  ", " ")
        cleaned.append(sentence.strip())
    return cleaned


class Vocab:
    def __init__(self, dataset_file, freq_threshold, dataset_name="coco"):
        self.dataset_file = dataset_file
        self.itos = {1: "<pad>", 2: "<bos>", 3: "<eos>", 0: "<unk>"}
        self.stoi = {"<pad>": 1, "<bos>": 2, "<eos>": 3, "<unk>": 0}
        self.freq_threshold = freq_threshold
        self.talk_file_location = f"data/{dataset_name}_talk.json"

        if os.path.exists(self.talk_file_location):
            self.load_vocabulary()
        else:
            self.build_vocabulary()
            self.load_vocabulary()

    def __len__(self):
        return len(self.itos)

    def load_vocabulary(self):
        with open(self.talk_file_location, "r") as f:
            self.itos = json.load(f)
            self.stoi = {v: int(k) for k, v in self.itos.items()}
        print(f"Loaded dictionary with {len(self.itos.items())} words")
        return

    def build_vocabulary(self):
        with open(self.dataset_file, "r") as f:
            karpathy_split = json.load(f)

        captions = []
        for image_data in karpathy_split["images"]:
            caps = [
                " ".join(sentence["tokens"]) for sentence in image_data["sentences"]
            ]
            captions.extend(caps)

        print(len(captions))

        caption_dictionary = {}
        for caption in captions:
            for word in caption.split(" "):
                caption_dictionary[word] = caption_dictionary.get(word, 0) + 1

        limited_caption_dictionary = {}
        for k, v in caption_dictionary.items():
            if v >= self.freq_threshold:
                limited_caption_dictionary[k] = v

        for i, k in enumerate(limited_caption_dictionary.keys()):
            self.stoi[k] = i + 4

        self.itos = {v: k for k, v in self.stoi.items()}

        # write self.itos to a json file
        os.mkdir("data") if not os.path.exists("data") else None
        with open(self.talk_file_location, "w+") as f:
            json.dump(self.itos, f)

    def numericalize(self, text):
        # text is a string
        # we want to return a list of integers
        # providing a numericalized version of the text
        tokenized_text = text.split(" ")
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]


# Adapted to give BU-TD features
class CaptioningDatasetWithFeatures(Dataset):
    def __init__(
        self,
        root_dir: str,
        captions_file: str,
        dataset_name: str = "coco",
        freq_threshold: int = 5,
        split: str = "train",
        feature_limit: int = 50,
        input_feature_size: int = 2048,
    ):
        self.root_dir = root_dir
        self.captions_file = captions_file
        self.freq_threshold = freq_threshold
        assert split in [
            "train",
            "val",
            "test",
        ], f"Split must be train, val or test. Received: {split}"
        self.split = split
        self.feature_limit = feature_limit
        self.input_feature_size = input_feature_size

        with open(self.captions_file, "r") as f:
            self.captions_file_data = json.load(f)

        self.image_locations = []
        self.captions = []

        for image_data in self.captions_file_data["images"]:
            if image_data["split"] == "restval":
                image_data["split"] = "train"

            if image_data["split"] == self.split:
                img_path = os.path.join(
                    self.root_dir,
                    f"{image_data['cocoid']}.npz"
                    if dataset_name == "coco"
                    else image_data["filename"].split(".")[0] + ".npz",
                )

                caps = [
                    " ".join(sentence["tokens"]) for sentence in image_data["sentences"]
                ]
                self.image_locations.append(img_path)
                self.captions.append(caps)

        self.length = len(self.captions)
        self.vocab = Vocab(freq_threshold, dataset_name)

    def __getitem__(self, index):
        img_path = self.image_locations[index]
        image = np.load(img_path)["feat"]
        image = torch.from_numpy(image).type(torch.float32)

        # if image.shape[0] < self.feature_limit then we need to pad it with zeros
        if image.shape[0] < self.feature_limit:
            pad = torch.zeros(
                (self.feature_limit - image.shape[0], self.input_feature_size)
            )
            image = torch.cat((image, pad), 0)

        captions = self.captions[index]

        # randomly select a caption from the list of captions
        caption = "<bos> " + np.random.choice(captions) + " <eos>"
        seq = self.vocab.numericalize(caption)

        return image[: self.feature_limit], seq, captions

    def __len__(self):
        return self.length

    @property
    def text(self):
        return list(chain.from_iterable(caption_list for caption_list in self.captions))

    def decode(self, word_idxs, join_words=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode(
                [
                    word_idxs,
                ],
                join_words,
            )[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode(
                [
                    word_idxs,
                ],
                join_words,
            )[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            for wi in wis:
                word = self.vocab.itos[str(wi.item())]
                if word == "<eos>":
                    break
                if word == "<bos>":
                    continue
                caption.append(word)
            if join_words:
                caption = " ".join(caption)
            captions.append(caption)
        return captions


class CaptioningDatasetFromHfpy(Dataset):
    def __init__(
        self,
        h5py_file: str,
        captions_file: str,
        dataset_name: str = "coco",
        freq_threshold: int = 5,
        split: str = "train",
        feature_limit: int = 50,
        input_feature_size: int = 2048,
    ):
        self.h5py_file = h5py_file
        self.captions_file = captions_file
        self.freq_threshold = freq_threshold
        assert split in [
            "train",
            "val",
            "test",
        ], f"Split must be train, val or test. Received: {split}"
        self.split = split
        self.feature_limit = feature_limit
        self.input_feature_size = input_feature_size

        with open(self.captions_file, "r") as f:
            self.captions_file_data = json.load(f)

        self.image_locations = []
        self.captions = []

        for image_data in self.captions_file_data["images"]:
            if image_data["split"] == "restval":
                image_data["split"] = "train"

            if image_data["split"] == self.split:
                self.image_locations.append(image_data["cocoid"])

                caps = [
                    " ".join(sentence["tokens"]) for sentence in image_data["sentences"]
                ]
                self.captions.append(caps)

        self.length = len(self.captions)
        self.vocab = Vocab(captions_file, freq_threshold, dataset_name)

    def __getitem__(self, index):
        img_id = self.image_locations[index]
        with h5py.File(self.h5py_file, "r") as f:
            image = f[f"{img_id}_features"][()]
        image = torch.from_numpy(image).type(torch.float32)

        # if image.shape[0] < self.feature_limit then we need to pad it with zeros
        if image.shape[0] < self.feature_limit:
            pad = torch.zeros(
                (self.feature_limit - image.shape[0], self.input_feature_size)
            )
            image = torch.cat((image, pad), 0)

        captions = self.captions[index]

        # randomly select a caption from the list of captions
        # caption = "<bos> " + np.random.choice(captions) + " <eos>"
        caption = "<bos> " + captions[0] + " <eos>"
        seq = self.vocab.numericalize(caption)

        return image[: self.feature_limit], seq, captions

    def __len__(self):
        return self.length

    @property
    def text(self):
        return list(chain.from_iterable(caption_list for caption_list in self.captions))

    def decode(self, word_idxs, join_words=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode(
                [
                    word_idxs,
                ],
                join_words,
            )[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode(
                [
                    word_idxs,
                ],
                join_words,
            )[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            for wi in wis:
                word = self.vocab.itos[str(wi.item())]
                if word == "<eos>":
                    break
                # if word == "<bos>":
                #     continue
                caption.append(word)
            if join_words:
                caption = " ".join(caption)
            captions.append(caption)
        return captions


class DuplicatedCaptioningDatasetFromHfpy(Dataset):
    def __init__(
        self,
        h5py_file: str,
        captions_file: str,
        dataset_name: str = "coco",
        freq_threshold: int = 5,
        split: str = "train",
        feature_limit: int = 50,
        input_feature_size: int = 2048,
    ):
        self.h5py_file = h5py_file
        self.captions_file = captions_file
        self.freq_threshold = freq_threshold
        assert split in [
            "train",
            "val",
            "test",
        ], f"Split must be train, val or test. Received: {split}"
        self.split = split
        self.feature_limit = feature_limit
        self.input_feature_size = input_feature_size

        with open(self.captions_file, "r") as f:
            self.captions_file_data = json.load(f)

        self.image_locations = []
        self.captions = []
        self.training_captions = []

        for image_data in self.captions_file_data["images"]:
            if image_data["split"] == "restval":
                image_data["split"] = "train"

            if image_data["split"] == self.split:
                for i in range(5):
                    self.image_locations.append(image_data["cocoid"])
                    caps = [
                        " ".join(sentence["tokens"])
                        for sentence in image_data["sentences"]
                    ]
                    self.captions.append(caps)
                    self.training_captions.append(caps[i])

        self.length = len(self.captions)
        self.vocab = Vocab(captions_file, freq_threshold, dataset_name)

    def __getitem__(self, index):
        img_id = self.image_locations[index]
        with h5py.File(self.h5py_file, "r") as f:
            image = f[f"{img_id}_features"][()]
        image = torch.from_numpy(image).type(torch.float32)

        # if image.shape[0] < self.feature_limit then we need to pad it with zeros
        if image.shape[0] < self.feature_limit:
            pad = torch.zeros(
                (self.feature_limit - image.shape[0], self.input_feature_size)
            )
            image = torch.cat((image, pad), 0)

        captions = self.captions[index]

        caption = "<bos> " + self.training_captions[index] + " <eos>"
        seq = self.vocab.numericalize(caption)

        return image[: self.feature_limit], seq, captions

    def __len__(self):
        return self.length

    @property
    def text(self):
        return list(chain.from_iterable(caption_list for caption_list in self.captions))

    def decode(self, word_idxs, join_words=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode(
                [
                    word_idxs,
                ],
                join_words,
            )[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode(
                [
                    word_idxs,
                ],
                join_words,
            )[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            for wi in wis:
                word = self.vocab.itos[str(wi.item())]
                if word == "<eos>":
                    break
                # if word == "<bos>":
                #     continue
                caption.append(word)
            if join_words:
                caption = " ".join(caption)
            captions.append(caption)
        return captions


class Batcher:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, batch):
        images, seq, captions = zip(*batch)
        images = torch.stack(images, 0)
        max_len = max(len(s) for s in seq)

        seq_padded = []
        for s in seq:
            padded = s + [self.vocab.stoi["<pad>"]] * (max_len - len(s))
            seq_padded.append(padded)

        seq_padded = torch.LongTensor(seq_padded)

        return images, seq_padded, captions
