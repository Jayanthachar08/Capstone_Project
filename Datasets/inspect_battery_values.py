
import h5py
import numpy as np

FILE_PATH = "2017-05-12_batchdata_updated_struct_errorcorrect.mat"


def read_scalar(file, dataset, index):
    ref = dataset[index, 0]
    values = np.asarray(file[ref]).ravel()
    return float(values[0])


with h5py.File(FILE_PATH, "r") as f:
    batch = f["batch"]
    summary = batch["summary"]
    cycle_life = batch["cycle_life"]

    print("FIRST 5 BATTERIES\n")

    for i in range(min(5, summary.shape[0])):
        summary_ref = summary[i, 0]
        battery_summary = f[summary_ref]

        capacity = np.asarray(
            battery_summary["QDischarge"]
        ).ravel()

        cycles = np.asarray(
            battery_summary["cycle"]
        ).ravel()

        life = read_scalar(f, cycle_life, i)

        print(f"Battery {i + 1}")
        print(f"Recorded cycle life: {life}")
        print(f"Number of summary records: {len(capacity)}")
        print(f"First cycle number: {cycles[0]}")
        print(f"Last cycle number: {cycles[-1]}")

        print("First 10 discharge capacities:")
        print(np.round(capacity[:10], 5))

        print("Last 10 discharge capacities:")
        print(np.round(capacity[-10:], 5))

        print(f"Minimum capacity: {np.min(capacity):.5f}")
        print(f"Maximum capacity: {np.max(capacity):.5f}")
        print(f"Median capacity: {np.median(capacity):.5f}")
        print(f"Zero capacities: {np.sum(capacity == 0)}")
        print(f"Invalid capacities: {np.sum(~np.isfinite(capacity))}")
        print("-" * 45)
