import numpy as np
import h5py


def npz2h5(npz_file, h5_file):
    npz = np.load(npz_file)

    with h5py.File(h5_file, 'w') as h5:
        for key in npz.files:
            arr = npz[key]
            if arr.dtype.kind in {'U', 'S'}:  # Handle string arrays
                dt = h5py.string_dtype(encoding='utf-8')
                h5.create_dataset(key, data=arr.astype(object), dtype=dt)
            else:
                h5.create_dataset(key, data=arr)
