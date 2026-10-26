#!/usr/bin/env python3
"""Train SAR-YOLO26 with the verified reproduction protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("m",), default="m")
    parser.add_argument("--dataset", choices=("sar-aircraft",), default="sar-aircraft")
    parser.add_argument("--model", help="Optional custom architecture YAML")
    parser.add_argument("--data", help="Optional custom dataset YAML")
    parser.add_argument("--pretrained", help="Defaults to yolo26m.pt; 'none' trains from scratch")
    parser.add_argument("--epochs", type=int, help="Defaults to 150 for SAR-AIRcraft-1.0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=str(ROOT / "runs/train"))
    parser.add_argument("--name")
    parser.add_argument("--omega1", type=float, default=1.0, help="QG-CIoU CIoU weight")
    parser.add_argument("--omega2", type=float, default=0.20, help="QG-CIoU axis-offset weight")
    parser.add_argument("--omega3", type=float, default=0.05, help="QG-CIoU logarithmic size weight")
    parser.add_argument("--without-lcad", action="store_true", help="Table III: use baseline stride-2 Conv")
    parser.add_argument("--without-apdr", action="store_true", help="Tables III/IV: use baseline C3k2")
    parser.add_argument("--without-qg-ciou", action="store_true", help="Table III: use baseline CIoU")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without training")
    parser.add_argument("--stop-after", type=int, help="Stop after N epochs while retaining the full --epochs schedule")
    parser.add_argument("--allow-env-mismatch", action="store_true", help="Explicitly allow training outside the pinned reproduction environment")
    args = parser.parse_args(argv)
    if min(args.omega1, args.omega2, args.omega3) < 0:
        parser.error("QG-CIoU weights must be nonnegative")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    return args


def resolve_settings(args):
    model_file = args.model or str(ROOT / "configs/models" / f"sar_yolo26{args.variant}.yaml")
    data_name = "sar_aircraft"
    epochs = args.epochs if args.epochs is not None else 150
    settings = dict(
        data=args.data or str(ROOT / f"configs/data/{data_name}.yaml"),
        epochs=epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        workers=args.workers, project=args.project,
        name=args.name or f"cdpr_yolo26{args.variant}_{data_name}_paper",
        seed=0, deterministic=False, optimizer="auto",
        lr0=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
        warmup_bias_lr=0.0,
        patience=100, close_mosaic=10,
        box_iou_type="ciou" if args.without_qg_ciou else "qgciou",
        qgciou_ciou_w=args.omega1, qgciou_axis_w=args.omega2, qgciou_shape_w=args.omega3,
        wiou_gamma=1.9, wiou_delta=3.0, wiou_half_life=7000.0,
    )
    return model_file, args.pretrained or f"yolo26{args.variant}.pt", settings


def architecture(model_file, args):
    config = yaml.safe_load(Path(model_file).read_text())
    for section in ("backbone", "head"):
        for layer in config[section]:
            if args.without_lcad and layer[2] in {"LCAD", "LSConv"}:
                layer[2], layer[3] = "Conv", layer[3][:3]
            if args.without_apdr and layer[2] in {"APDR", "C3k2_LWGA_1"}:
                layer[2] = "C3k2"
    return config


def main() -> None:
    args = parse_args()
    model_file, pretrained, settings = resolve_settings(args)
    config = architecture(model_file, args)
    if args.dry_run:
        print(json.dumps(dict(model=model_file, pretrained=pretrained, architecture=config,
                              train=settings), indent=2))
        return
    from scripts.runtime_env import environment_snapshot, validate_environment
    runtime = environment_snapshot()
    differences = validate_environment(runtime, allow_mismatch=args.allow_env_mismatch)
    print("Training environment: " + json.dumps(runtime, ensure_ascii=False), flush=True)
    if differences:
        print("Intentional environment mismatch: " + json.dumps(differences), flush=True)
    # DDP starts scripts outside the repository; propagate this fork explicitly.
    search_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), *[p for p in search_paths if p and p != str(ROOT)]])
    model = YOLO(model_file)
    if args.without_lcad or args.without_apdr:
        from ultralytics.nn.tasks import DetectionModel
        model.model = DetectionModel(config, verbose=True)
    if pretrained.lower() != "none":
        model.load(pretrained)
    if args.stop_after is not None:
        from scripts.short_run import PrefixDetectionTrainer
        os.environ["CDPR_PREFIX_EPOCHS"] = str(args.stop_after)
        model.train(trainer=PrefixDetectionTrainer, **settings)
    else:
        model.train(**settings)


if __name__ == "__main__":
    main()

