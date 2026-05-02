# PPE Detection System — CS7002NU Assessment 1

Computer vision system for London Building Materials Manufacturing Ltd. that detects
PPE compliance (helmets, hi-vis vests, goggles) from factory camera feeds.

## Architecture

| Layer | Tool | Purpose |
|-------|------|---------|
| Code | GitHub | Clean, versioned Python modules |
| Compute | Google Colab (T4 GPU) | Training, evaluation, experiments |
| Storage | Google Drive | Datasets, checkpoints, outputs |

## Project Structure

```
ai-vision-ppe-detection/
├── src/
│   ├── section_a/image_ops.py      # Section A: OpenCV image processing
│   ├── section_b/dataset.py        # YOLO -> TFRecord conversion
│   ├── section_b/model.py          # EfficientDet-D0 setup via TF OD API
│   ├── section_b/trainer.py        # Training launcher (3 configs)
│   ├── section_b/evaluate.py       # mAP, precision, recall, F1
│   └── utils/visualise.py          # Bbox drawing, plot helpers
├── config/
│   ├── hyperparams.py              # 3 hyperparameter configurations
│   └── pipeline_template.config    # EfficientDet-D0 pipeline template
└── PPE_Detection.ipynb  # Single submission notebook (Sec A + B)   
```

## Google Drive Layout

```
MyDrive/CS7002NU_PPE/
├── datasets/raw/          <- YOLO dataset (upload from local Datasets/ folder)
├── datasets/tfrecords/    <- Generated TFRecords (train / val / test)
├── checkpoints/           <- EfficientDet-D0 COCO17 base checkpoint
├── models/config1|2|3/    <- Training outputs per hyperparameter config
├── exports/saved_model/   <- Final exported SavedModel (best config)
└── outputs/section_a|b/   <- Result images and evaluation plots
```

## Team

| Role | Owner | Branch |
|------|-------|--------|
| Lead / Section A | S1 | `feature/section-a` |
| Data Pipeline | S2 | `feature/dataset-pipeline` |
| Model Setup | S3 | `feature/model-setup` |
| Evaluation + LSEP | S4 | `feature/evaluation` |

## Quick Start (Google Colab)

1. Open `PPE_Detection.ipynb` in Google Colab
2. Set runtime to **GPU (T4)**: Runtime -> Change runtime type -> T4 GPU
3. Run the **Setup** cell — mounts Drive, clones repo, installs TF OD API
4. Execute sections in order (guards skip steps already completed on Drive)

## Submission Files

- `CS7002NU_PPE_Detection.ipynb` — downloaded from Colab
- `LMU_ID_Number_CS7002NU_A1_Report.pdf` — technical report
- `LMU_ID_Number_CS7002NU_A1_LSEP.pdf` — LSEP essay
- Colab share link (File -> Share -> Anyone with the link)
