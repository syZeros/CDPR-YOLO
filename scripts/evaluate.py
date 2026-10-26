#!/usr/bin/env python3
"""Generate a checkpoint metrics report (paper mAP uses IoU 0.50:0.95)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a checkpoint metrics report.")
    parser.add_argument(
        "--weights",
        default=str(ROOT / "runs/train/cdpr_yolo26m_sar_aircraft_paper/weights/best.pt"),
        help="Checkpoint path; defaults to the output of the standard M training command",
    )
    parser.add_argument("--data", default=str(ROOT / "configs/data/sar_aircraft.yaml"),
                        help="Dataset YAML; defaults to configs/data/sar_aircraft.yaml")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", help="Optional path for machine-readable metric JSON")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    summary = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP75": float(metrics.box.map75),
        "mAP50-95": float(metrics.box.map),
    }
    # Mean per-class F1 at the validator-selected operating point. This is
    # distinct from taking the harmonic mean of macro precision and recall.
    summary["F1"] = float(metrics.box.f1.mean())
    summary["params_m"] = get_num_params(model.model) / 1e6
    summary["gflops"] = get_flops(model.model, imgsz=args.imgsz)
    summary["per_class"] = {
        model.names[int(class_id)]: {
            "precision": float(metrics.box.p[index]),
            "recall": float(metrics.box.r[index]),
            "F1": float(metrics.box.f1[index]),
            "mAP50-95": float(metrics.box.all_ap[index].mean()),
            "mAP50": float(metrics.box.all_ap[index, 0]),
            "mAP75": float(metrics.box.all_ap[index, 5]),
        }
        for index, class_id in enumerate(metrics.box.ap_class_index)
    }
    summary["speed_ms"] = {key: float(value) for key, value in metrics.speed.items()}
    summary["weights"] = args.weights
    summary["data"] = args.data
    summary["split"] = args.split
    summary["imgsz"] = args.imgsz
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n")
    for name, value in summary.items():
        if isinstance(value, dict):
            print(f"{name}: {json.dumps(value, ensure_ascii=False)}")
        else:
            print(f"{name}: {value:.6f}" if isinstance(value, float) else f"{name}: {value}")


if __name__ == "__main__":
    main()

