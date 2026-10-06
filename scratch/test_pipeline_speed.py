import sys
import time
from pathlib import Path
import pandas as pd

sys.path.insert(0, "src")
from cxr_reliability.dashboard.app import load_pipeline

pipeline, thresholds, active_path, info = load_pipeline()
df_test = pd.read_csv("data/processed/test.csv")
pneu_samples = df_test[df_test["label"] == 1].head(3)
non_pneu_samples = df_test[df_test["label"] == 0].head(3)
samples = pd.concat([pneu_samples, non_pneu_samples])

t0 = time.time()
for _, row in samples.iterrows():
    img_path = Path("dataset") / row["image_path"].replace("\\", "/")
    res = pipeline.run(img_path, input_id=row["image_id"])
    bm_before = res.base_model.raw_pneumonia_score if res.base_model else None
    bm_after = res.after_repair_base_model.raw_pneumonia_score if res.after_repair_base_model else None
    q_before = res.quality.overall.value if res.quality else None
    q_after = res.after_repair_quality.overall.value if res.after_repair_quality else None
    lap_before = res.quality.laplacian_variance if res.quality else None
    lap_after = res.after_repair_quality.laplacian_variance if res.after_repair_quality else None
    print(f"{row['image_id']}: q_before={q_before}, q_after={q_after}, lap_before={lap_before:.1f}, lap_after={lap_after if lap_after else 'N/A'}, bm_before={bm_before:.4f}, bm_after={bm_after if bm_after is not None else 'N/A'}, act={res.final_action.value}")
print(f"6 samples completed in {time.time() - t0:.2f}s")
