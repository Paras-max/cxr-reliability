# data/

Nothing in this directory except manifests is version-controlled.

| Path | Contents |
|---|---|
| `raw/` | Original downloads: NIH ChestX-ray14 (images + `Data_Entry_2017.csv`), CheXpert / PadChest samples used as natural OOD sets |
| `interim/` | Resized caches (224 and 512), temporary artifacts |
| `processed/` | Final model-ready arrays and the synthetic corruption sets (with severity metadata) |
| `manifests/` | Split manifests (patient-level) with content hashes; referenced by hash in every result |

Rules
- Public research datasets only. Never place real patient data here.
- Splits are grouped by patient ID; the test split is touched only after thresholds are frozen.
- CheXpert and PadChest may require access approval; check each dataset's terms before download.
