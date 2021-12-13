import pandas as pd
import numpy as np
import itertools
import sys

def read_fasta_file():
    '''
    used for load fasta data and transformd into numpy.array format
    '''
    fh = open('C:/Users/GWC/Desktop/bty668_supp/Additional file 1/Datasets 1/Positive_A1978.txt', 'r')
    seq = []
    for line in fh:
        if line.startswith('>'):
            continue
        else:
            seq.append(line.replace('\n', '').replace('\r', ''))
    fh.close()
    matrix_data = np.array([list(e) for e in seq])
    print(matrix_data)
    return matrix_data

def fetch_singleline_features_withoutN(sequence):
    alphabet="ACGT"
    k_num=2
    two_sequence=[]
    for index,data in enumerate(sequence):
        if index <(len(sequence)-k_num+1):
            two_sequence.append("".join(sequence[index:(index+k_num)]))
    parameter=[e for e in itertools.product([0,1],repeat=4)]
    record=[0 for x in range(int(pow(4,k_num)))]
    matrix=["".join(e) for e in itertools.product(alphabet, repeat=k_num)] # AA AU AC AG UU UC ...
    final=[]
    for index,data in enumerate(two_sequence):
        if data in matrix:
            final.extend(parameter[matrix.index(data)])
            record[matrix.index(data)]+=1
            final.append(record[matrix.index(data)]*1.0/(index+1))
    return final

matrix_data=read_fasta_file()
features_data=[]
for index,sequence in enumerate(matrix_data):
    features_data.append(fetch_singleline_features_withoutN(sequence))
print(np.array(features_data).shape)

sequence = matrix_data[0]
alphabet = "ACGT"
k_num = 2
two_sequence = []
for index, data in enumerate(sequence):
    if index < (len(sequence) - k_num + 1):
        two_sequence.append("".join(sequence[index:(index + k_num)]))
parameter = [e for e in itertools.product([0, 1], repeat=4)]
record = [0 for x in range(int(pow(4, k_num)))]
matrix = ["".join(e) for e in itertools.product(alphabet, repeat=k_num)]  # AA AU AC AG UU UC ...
final = []
for index, data in enumerate(two_sequence):
    if data in matrix:
        final.extend(parameter[matrix.index(data)])
        record[matrix.index(data)] += 1
        final.append(record[matrix.index(data)] * 1.0 / (index + 1))

pd.DataFrame(features_data).to_csv('Positive_A1978_dbpf.csv',header=None,index=False)