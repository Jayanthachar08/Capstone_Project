
import h5py

file_path = r"C:\engg notes\7th sem\data\2017-05-12_batchdata_updated_struct_errorcorrect.mat"

with h5py.File(file_path, "r") as f:
    batch = f["batch"]

    for field in ["barcode", "cycle_life", "summary", "cycles"]:
        print(f"\n{'=' * 20} {field.upper()} {'=' * 20}")

        dataset = batch[field]
        print("Shape:", dataset.shape)
        print("Data type:", dataset.dtype)

        # Show the first three stored references
        for i in range(min(3, dataset.shape[0])):
            ref = dataset[i, 0]

            if not ref:
                print(f"Record {i}: Empty reference")
                continue

            obj = f[ref]

            if isinstance(obj, h5py.Group):
                print(f"\nRecord {i}: Group")
                for name, child in obj.items():
                    if isinstance(child, h5py.Dataset):
                        print(
                            f"  {name}: shape={child.shape}, "
                            f"type={child.dtype}"
                        )
                    else:
                        print(f"  {name}: group")

            elif isinstance(obj, h5py.Dataset):
                print(
                    f"Record {i}: Dataset, "
                    f"shape={obj.shape}, type={obj.dtype}"
                )
