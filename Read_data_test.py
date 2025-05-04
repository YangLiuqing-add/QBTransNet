from scipy import stats as sci
import numpy as np
import obspy
from Process import gen_scalograms
import h5py

# ========= Load Earthquake events (labelled 0) =========
eids_eq = np.load('./Data/All_SWDdp.npy')
Neq = 270

def read_earthquake(eid):
    st = obspy.read(f'C:/Users/liuya444/PycharmProjects/Postdoc/QBNet/TexNet/waveforms/{eid}.mseed')
    cat = obspy.read_events(f'C:/Users/liuya444/PycharmProjects/Postdoc/QBNet/TexNet/events/{eid}.qml')
    return cat, st

data_eq = []
ii = 211
for ie in eids_eq[ii:Neq]:
    print(f'[EQ] Processing Event: {ie}  ({ii+1}/{Neq})')
    cat, st = read_earthquake(ie)
    datfin = gen_scalograms(cat, st)
    data_eq.append(datfin)
    print(f'Shape: {datfin.shape}')
    ii += 1

data_eq = np.concatenate(data_eq, axis=0)
label_eq = np.zeros(data_eq.shape[0])
print(f'[EQ] Data shape: {data_eq.shape}')


# ========= Load Quarry Blast events (labelled 1) =========
eids_qb = []
with open('./Data/eveQB_removed.dat') as f:
    lines = f.readlines()
    for line in lines:
        eids_qb.append(line.strip())

Nqb = 680

def read_quarryblast(eid):
    st = obspy.read(f'C:/Users/liuya444/PycharmProjects/Postdoc/QBNet/Data/quarryblast_waveforms/{eid}.mseed')
    cat = obspy.read_events(f'C:/Users/liuya444/PycharmProjects/Postdoc/QBNet/Data/quarryblast_events/{eid}.qml')
    return cat, st

data_qb = []
ii = 531
for ie in eids_qb[ii:Nqb]:
    print(f'[QB] Processing Event: {ie}  ({ii+1}/{Nqb})')
    cat, st = read_quarryblast(ie)
    datfin = gen_scalograms(cat, st)
    if ie not in ['texnet2018nzam', 'texnet2018mmsh']:
        if datfin.shape[0] != 0:
            data_qb.append(datfin)
        else:
            print(f"[QB] Skipped empty event: {ie}")
    print(f'Shape: {datfin.shape}')
    ii += 1

data_qb = np.concatenate(data_qb, axis=0)
label_qb = np.ones(data_qb.shape[0])  # QB labeled as 1
print(f'[QB] Data shape: {data_qb.shape}')


# ========= Merge and Save dataset =========
datall = np.concatenate([data_eq, data_qb], axis=0)
laballx = np.concatenate([label_eq, label_qb], axis=0)

print(f'[ALL] Combined data shape: {datall.shape}')
print(f'[ALL] Combined label shape: {laballx.shape}')
print(f'[ALL] Label distribution (EQ=0, QB=1): {np.unique(laballx, return_counts=True)}')

with h5py.File('./Data/data_test.h5', 'w') as hf:
    hf.create_dataset('datall', data=datall)
    hf.create_dataset('laballx', data=laballx)

