import json
import os
import numpy as np
import pandas as pd
import torch

with open('kmers.json', 'r') as f:
    kmer_to_index = json.load(f)

def load_data(directory):
    data = []
    labels = []
    k_value = 3
    for label, category in enumerate(['crc', 'hc']):
        path = os.path.join(directory, category)
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            # extract the CDR3 sequences and counts, encode the sequences,
            # and normalize the counts.
            df = pd.read_csv(file_path, sep='\t')
            df = df.head(10)
            sequences = df['aaSeqCDR3'].values  # Replace 'CDR3' with the actual column name.
            counts = df['cloneCount'].values  # Replace 'counts' with the actual column name.
            # TODO: Encode the sequences and normalize the counts.
            encoded_sequences = encode_sequences_kmer(sequences, counts, k_value)
            data.append(encoded_sequences)
            labels.append(label)
    return data, labels

def kmer_count(sequence, k):
    kmer_dict = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer in kmer_dict:
            kmer_dict[kmer] += 1
        else:
            kmer_dict[kmer] = 1
    return kmer_dict

def encode_sequences_kmer(sequences, counts, k_value):
    num_kmers = len(kmer_to_index)
    # 为每个序列初始化一个零向量
    encoded_sequences = np.zeros((len(sequences), num_kmers))
    # 为每个 k-mer 分配一个索引

    # 为每个序列计算加权的 k-mer 频率
    for i, (seq, count) in enumerate(zip(sequences, counts)):
        kmer_freqs = kmer_count(seq, k_value)
        for kmer, freq in kmer_freqs.items():
            if kmer in kmer_to_index:
                encoded_sequences[i, kmer_to_index[kmer]] += freq * count
    # 归一化 k-mer 频率
    # 计算总的加权克隆计数
    total_counts = np.sum(counts)
    encoded_sequences /= total_counts
    return encoded_sequences