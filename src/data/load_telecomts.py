# src/data/load_telecomts.py
from datasets import load_dataset

def load_telecomts(dataset_name: str, data_files_pattern: str, offline_ok: bool = True):
    """
    Robust loader for TelecomTS.

    offline_ok=True:
      - if HF Hub is unreachable or dataset not found, try to reuse local cache.
    """

    # (définition) download_mode:
    # - "reuse_dataset_if_exists" = utilise le cache local si possible, sinon télécharge
    # - "force_redownload" = redownload complet (pas ce qu'on veut ici)
    download_mode = "reuse_dataset_if_exists"

    try:
        ds = load_dataset(
            dataset_name,
            data_files={"data": data_files_pattern},
            download_mode=download_mode,
        )
        return ds["data"]

    except Exception as e:
        if not offline_ok:
            raise

        # fallback: try loading without data_files (sometimes cache config differs)
        try:
            ds = load_dataset(dataset_name, download_mode=download_mode)
            return ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        except Exception:
            # raise original error with context
            raise RuntimeError(
                f"Failed to load dataset '{dataset_name}'. "
                f"HF may be unreachable and local cache config mismatched. "
                f"Original error: {repr(e)}"
            )
