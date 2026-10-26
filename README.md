# CDPR-YOLO: SAR Aircraft Detection

Implementation accompanying **Context-Aware Downsampling and Pixel-Directional Refinement Network for SAR Aircraft Detection**.

## Overview

CDPR-YOLO is an end-to-end detector built on YOLO26 for aircraft detection in synthetic aperture radar images. The S and M variants retain the scaling settings of YOLO26s and YOLO26m.



## Project Structure

```text
.
├── ultralytics/              # Core implementation
├── configs/                  # Model and dataset settings
├── mytrain.py                # Training
├── myreport.py               # Metrics report
└── requirements.txt
```

## Installation

Base environment: Python 3.10.14, PyTorch 2.4.1+cu121, torchvision 0.19.1+cu121, CUDA runtime 12.1, and cuDNN 9.1.0. The experiments use two NVIDIA RTX 3090 GPUs.

```bash
conda env create -f environment.yml
conda activate cdpr-yolo-repro
python scripts/smoke_test.py
```

## Datasets

| Dataset | Images | Classes | Paper protocol | Configuration |
|---|---:|---:|---|---|
| SAR-AIRcraft-1.0 | 4,368 | 7 | Publisher's split; 150 epochs | [sar_aircraft.yaml](configs/data/sar_aircraft.yaml) |
| SADD | 2,966 | 1 | 8:1:1; 100 epochs | [sadd.yaml](configs/data/sadd.yaml) |
| ZKXT | 2,000 | 7 | 8:1:1; 100 epochs | [zkxt.yaml](configs/data/zkxt.yaml) |

Download data from the original publishers and set an **absolute** `path` in each YAML. 


## Usage

### Training

```bash
# M / SAR-AIRcraft-1.0 / 150 epochs
# Download yolo26m.pt and place it in the repository root first.
python mytrain.py --variant m --dataset sar-aircraft \
  --pretrained yolo26m.pt --epochs 150 --imgsz 640 --batch 16 \
  --device 0,1 --project runs/train \
  --name cdpr_yolo26m_sar_aircraft_paper
```

### Metrics report

After training finishes, run the metrics report:

```bash
python myreport.py --output runs/evaluation/metrics.json
```

## Acknowledgments

This project builds on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) . We thank the SAR-AIRcraft-1.0, SADD, and ZKXT dataset authors.




