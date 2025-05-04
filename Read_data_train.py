from scipy import stats as sci
import numpy as np
import obspy
from Process import gen_scalograms
import h5py

### In the original paper, I marked EQ and QB as 1 and 0 respectively.
# When calculating Acc, Pre, Recall and F1-score, I manually switched the positive and negative classes back to QB as 0 and EQ as 1.
# For standardization, I marked EQ and QB as 0 and 1 respectively here.

# ========= Load Earthquake event （labelled 0） =========
eids_eq = np.load('./Dataset/All_SWDdp.npy')
Neq = 210

def read_earthquake(eid):
    st = obspy.read(f'C:/Users/liuya444/PycharmProjects/Postdoc/QBNet/TexNet/waveforms/{eid}.mseed')
    cat = obspy.read_events(f'C:/Users/liuya444/PycharmProjects/Postdoc/QBNet/TexNet/events/{eid}.qml')
    return cat, st

data_eq = []
ii = 0
for ie in eids_eq[0:Neq]:
    print(f'[EQ] Processing Event: {ie}  ({ii+1}/{Neq})')
    cat, st = read_earthquake(ie)
    datfin = gen_scalograms(cat, st)
    ii += 1
    data_eq.append(datfin)
    print(f'Shape: {datfin.shape}')

data_eq = np.concatenate(data_eq, axis=0)
label_eq = np.zeros(data_eq.shape[0])
print(f'[EQ] Data shape: {data_eq.shape}')

# ========= Load Quarry Blast （labelled 1） =========
eids_qb = []
with open('./Dataset/eveQB_removed.dat') as f:
    lines = f.readlines()
    for line in lines:
        eids_qb.append(line.strip())

Nqb = 530

def read_quarryblast(eid):
    st = obspy.read(f'C:/Users/liuya444/PycharmProjects/Postdoc/QBNet/Data/quarryblast_waveforms/{eid}.mseed')
    cat = obspy.read_events(f'C:/Users/liuya444/PycharmProjects/Postdoc/QBNet/Data/quarryblast_events/{eid}.qml')
    return cat, st

data_qb = []
ii = 0
for ie in eids_qb[0:Nqb]:
    print(f'[QB] Processing Event: {ie}  ({ii+1}/{Nqb})')
    cat, st = read_quarryblast(ie)
    datfin = gen_scalograms(cat, st)
    ii += 1
    if ie not in ['texnet2018nzam', 'texnet2018mmsh']:
        if datfin.shape[0] != 0:
            data_qb.append(datfin)
        else:
            print(f"[QB] Skipped empty event: {ie}")
    print(f'Shape: {datfin.shape}')

data_qb = np.concatenate(data_qb, axis=0)
label_qb = np.ones(data_qb.shape[0])  # QB 标记为 1
print(f'[QB] Data shape: {data_qb.shape}')

# ========= Generate dataset =========
datall = np.concatenate([data_eq, data_qb], axis=0)
laballx = np.concatenate([label_eq, label_qb], axis=0)

print(f'[ALL] Combined data shape: {datall.shape}')
print(f'[ALL] Combined label shape: {laballx.shape}')
print(f'[ALL] Label distribution (EQ=0, QB=1): {np.unique(laballx, return_counts=True)}')

# ========= Save dataset =========
with h5py.File('./Dataset/data_train.h5', 'w') as hf:
    hf.create_dataset('datall', data=datall)
    hf.create_dataset('laballx', data=laballx)

