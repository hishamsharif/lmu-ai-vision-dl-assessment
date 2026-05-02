"""
evaluate.py  —  Task 6: Model Evaluation & Inference
-----------------------------------------------------
COCO evaluation, per-class metrics, and detection visualisation.

Implement:
    run_coco_eval(pipeline_config_path, checkpoint_dir, num_eval_steps)
    extract_per_class_metrics(eval_events_dir, class_names)
    plot_metrics_comparison(config_results, output_path)
    visualise_detections(saved_model_dir, test_images_dir,
                         label_map_path, output_dir, n_samples)
"""


def run_coco_eval(pipeline_config_path: str,
                  checkpoint_dir: str,
                  num_eval_steps: int = None) -> dict:
    pass


def extract_per_class_metrics(eval_events_dir: str,
                               class_names: list) -> object:
    pass


def plot_metrics_comparison(config_results: dict,
                             output_path: str) -> None:
    pass


def visualise_detections(saved_model_dir: str,
                          test_images_dir: str,
                          label_map_path: str,
                          output_dir: str,
                          n_samples: int = 9) -> None:
    pass
