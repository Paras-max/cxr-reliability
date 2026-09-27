import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cxr_reliability.dashboard.app import load_pipeline, FULL_OOD_STATS_PATH

pipeline, thresholds, device, ood_info = load_pipeline(str(FULL_OOD_STATS_PATH))
df_test = pd.read_csv("data/processed/test.csv", nrows=100)

quality_rank = {"poor": 0, "degraded": 1, "good": 2}

results = []

for idx, row in df_test.iterrows():
    img_id = str(row["image_id"])
    rel_path = str(row["image_path"]).replace("\\", "/")
    img_path = PROJECT_ROOT / "dataset" / rel_path
    
    res = pipeline.run(img_path, input_id=img_id)
    
    # Check decision history
    init_dec = res.decision_history[0] if res.decision_history else None
    init_rule = init_dec.rule_id if init_dec else None
    init_action = init_dec.action.value if init_dec else None
    
    # Check repair and verification
    repaired = bool(res.repair and (res.repair.repair_applied or res.repair.repaired))
    ver = res.verification
    
    q_before = res.quality.overall.value if res.quality else None
    q_after = res.after_repair_quality.overall.value if res.after_repair_quality else None
    
    u_before = res.uncertainty.uncertainty_level.value if res.uncertainty else None
    u_after = res.after_repair_uncertainty.uncertainty_level.value if res.after_repair_uncertainty else None
    
    raw_before = getattr(res.base_model, 'raw_pneumonia_score', getattr(res.base_model, 'pneumonia_probability', None)) if res.base_model else None
    raw_after = getattr(res.after_repair_base_model, 'raw_pneumonia_score', getattr(res.after_repair_base_model, 'pneumonia_probability', None)) if res.after_repair_base_model else None
    
    conf_before = ver.confidence_before if ver else (res.uncertainty.confidence if res.uncertainty else None)
    conf_after = ver.confidence_after if ver else None
    delta_conf = ver.delta_confidence if ver else None
    
    results.append({
        "image_id": img_id,
        "init_action": init_action,
        "init_rule": init_rule,
        "final_action": res.final_action.value,
        "repair_attempts": res.repair_attempts,
        "repair_applied": repaired,
        "quality_before": q_before,
        "quality_after": q_after,
        "quality_improved": (quality_rank.get(q_after, -1) > quality_rank.get(q_before, -1)) if (q_after and q_before) else False,
        "quality_worsened": (quality_rank.get(q_after, -1) < quality_rank.get(q_before, -1)) if (q_after and q_before) else False,
        "quality_unchanged": (quality_rank.get(q_after, -1) == quality_rank.get(q_before, -1)) if (q_after and q_before) else False,
        "conf_before": conf_before,
        "conf_after": conf_after,
        "delta_conf": delta_conf,
        "delta_conf_pos": (delta_conf > 0) if delta_conf is not None else False,
        "delta_conf_neg": (delta_conf < 0) if delta_conf is not None else False,
        "delta_conf_zero": (delta_conf == 0) if delta_conf is not None else False,
        "delta_conf_ge_15": (delta_conf >= 0.15) if delta_conf is not None else False,
        "ver_verified": ver.verified if ver else None,
        "ver_next_step": ver.next_step.value if ver else None,
        "ver_reasoning": ver.reasoning if ver else None,
    })

df_res = pd.DataFrame(results)
print("=== OVERALL BREAKDOWN ===")
print("Total rows:", len(df_res))
print("Initial actions:\n", df_res["init_action"].value_counts())
print("Initial rules:\n", df_res["init_rule"].value_counts())
print("Final actions:\n", df_res["final_action"].value_counts())

print("\n=== REPAIRED IMAGES (repair_applied == True, N=88) ===")
rep = df_res[df_res["repair_applied"] == True]
print("Count:", len(rep))
print("Quality improved:", rep["quality_improved"].sum())
print("Quality worsened:", rep["quality_worsened"].sum())
print("Quality unchanged:", rep["quality_unchanged"].sum())
print("Quality transitions:\n", rep.groupby(["quality_before", "quality_after"]).size())

print("\n=== CONFIDENCE DELTA (N=88) ===")
print("Positive (>0):", rep["delta_conf_pos"].sum())
print("Negative (<0):", rep["delta_conf_neg"].sum())
print("Zero (==0):", rep["delta_conf_zero"].sum())
print("Achieved delta >= 0.15:", rep["delta_conf_ge_15"].sum())
print("Min delta:", rep["delta_conf"].min())
print("Max delta:", rep["delta_conf"].max())
print("Mean delta:", rep["delta_conf"].mean())
print("Median delta:", rep["delta_conf"].median())

print("\n=== VERIFICATION FAILURE REASONS (N=88) ===")
print(rep["ver_reasoning"].value_counts().head(10))

print("\n=== DIRECT ESCALATE (repair_attempts == 0, N=9) ===")
dir_esc = df_res[df_res["repair_attempts"] == 0]
print(dir_esc[["image_id", "init_action", "init_rule", "quality_before", "conf_before"]])

print("\n=== REPAIR ATTEMPTED BUT NOT APPLIED (N=3) ===")
not_app = df_res[(df_res["repair_attempts"] > 0) & (df_res["repair_applied"] == False)]
print(not_app[["image_id", "init_action", "init_rule", "quality_before", "conf_before"]])
