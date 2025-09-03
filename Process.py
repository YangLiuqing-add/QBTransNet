import obspy
from obspy.signal.tf_misfit import cwt
import numpy as np


# fs = 100
# dt = 1/100
# f_min = 1
# f_max = 45

def gen_scalograms(cat, st, dt=1.0 / 100, f_min=1, f_max=45, no_scale=20):
    '''
    gen_scalogram: Generate scalograms for HFSWD classification

    Modified from Omar's initial version by YC Apr 9, 2023

    INPUT
    cat: catalog
    st:  3C waveform stream
    dt: sampling rate
    f_min: minimum frequency
    f_max: maximum frequency
    no_scale: number of scales

    OUTPUT
    datout: an numpy array with size [no_station,no_sample, 6000, 3]

    EXAMPLE
    from pylib.hfswd import gen_scalograms

    demos/test_35_HFSWDclass_model.py

    '''
    st = st.resample(100)
    st = st.merge(fill_value=0)
    st = st.filter(type='bandpass', freqmin=1.0, freqmax=45, corners=2, zerophase=True)
    st = st.taper(max_percentage=0.001, type='cosine', max_length=2)

    datout = []
    for i in cat[0].picks:
        dat = []
        if i.phase_hint == 'P':
            st2 = st.copy()
            st2 = st2.select(station=i.waveform_id.station_code)
            if len(st2) == 3:
                if len(st2[2].data) >= 6000:
                    t = obspy.UTCDateTime(i.time)
                    stall = st2.trim(t - 1, t + 59 - 0.01)
                    #					 print(stall)
                    if len(stall) == 3:
                        a0 = stall[0].data
                        a1 = stall[1].data
                        a2 = stall[2].data
                        if (len(a0) == 6000) and (len(a1) == 6000) and (len(a2) == 6000):
                            dat.append(a0)
                            dat.append(a1)
                            dat.append(a2)
                    else:
                        a0 = []
                        a1 = []
                        a2 = []
                else:
                    a0 = []
                    a1 = []
                    a2 = []
                if (len(a0) == 6000) and (len(a1) == 6000) and (len(a2) == 6000):
                    dat = np.array(dat)
                    temp0 = cwt(a0, dt, 10, f_min, f_max, nf=no_scale, wl='morlet')
                    temp0 = np.clip(np.abs(temp0)[-1::-1], 0, 100)
                    # temp0 = temp0[3:,:]
                    ma = np.max(np.abs(temp0))
                    if ma == 0:
                        ma = 1
                    temp0 = temp0 / ma

                    temp1 = cwt(a1, dt, 10, f_min, f_max, nf=no_scale, wl='morlet')
                    temp1 = np.clip(np.abs(temp1)[-1::-1], 0, 100)
                    # temp1 = temp1[3:,:]
                    ma = np.max(np.abs(temp1))
                    if ma == 0:
                        ma = 1
                    temp1 = temp1 / ma

                    temp2 = cwt(a2, dt, 10, f_min, f_max, nf=no_scale, wl='morlet')
                    temp2 = np.clip(np.abs(temp2)[-1::-1], 0, 100)
                    # temp2 = temp2[3:,:]
                    ma = np.max(np.abs(temp2))
                    if ma == 0:
                        ma = 1
                    temp2 = temp2 / ma

                    tmp = np.zeros((no_scale, 6000, 3))
                    tmp[:, :, 0] = np.abs(temp0[:, 0:6000])
                    tmp[:, :, 1] = np.abs(temp1[:, 0:6000])
                    tmp[:, :, 2] = np.abs(temp2[:, 0:6000])
                    datout.append(tmp)
    datout = np.array(datout)
    return datout






