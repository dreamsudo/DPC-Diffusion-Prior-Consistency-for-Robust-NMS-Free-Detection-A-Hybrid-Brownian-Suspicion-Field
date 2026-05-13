"""Config schema for the DPC-YOLO26 v3.3.0 control plane.

Loads JSON config files into a typed tree of dataclasses. Validates
required fields, resolves all paths relative to the staging/ directory.

v3.3.0 update: joint Phase 2 training (denoiser + YOLO26 head) and full
paper-strict hyperparameter set (lambda_match, lambda_entropy,
lambda_locstab, lambda_anchor, fixed probe schedule, etc.) are now
first-class config fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    coco_dir: str
    apricot_dir: str
    coco_val_dir: Optional[str] = None
    apricot_split: str = "test"
    apricot_val_fraction: float = 0.1
    num_workers: int = 8


@dataclass
class PrepConfig:
    probe_resolution: int = 128  # paper §5.9 default
    eval_resolution: int = 640   # YOLO26 inference resolution
    coco_annotations: Optional[str] = None  # path to instances_train2017.json


@dataclass
class DpcMathConfig:
    """Mirror of dpc.config.DPCConfig fields exposed through the orchestrator.

    The actual DPCConfig dataclass (in dpc/config.py) is the authoritative
    schema; this dataclass just collects what the JSON config carries and
    passes it through. Trainers construct a DPCConfig from this.
    """
    # §5.1 diffusion
    diffusion_steps: int = 1000
    beta_start: float = 1.0e-4
    beta_end: float = 0.02
    # §5.4 K-probe schedule
    n_probes: int = 8
    sigma_min: float = 0.05
    sigma_max: float = 0.5
    # §5.9 probe resolution
    probe_res: int = 128
    # §5.6 smoothing
    smoothing_sigma: float = 1.5
    smoothing_kernel_size: int = 7
    # §5.8 deployment selection
    deployment_mode: str = "hybrid"
    fusion_weight_l1: float = 0.5
    # §5.11, §5.12 calibration
    lambda_obj: float = 50.0
    lambda_cls: float = 50.0
    class_calibration_mode: str = "uniform"
    # §5.15 small-target amplification
    lambda_small: float = 0.5
    small_target_area_threshold: float = 0.01
    # §5.16 assignment-cost modulation (training only)
    lambda_match: float = 10.0
    # §5.13, §5.14 auxiliary losses
    lambda_entropy: float = 0.1
    lambda_locstab: float = 0.0
    # YOLO26 head fine-tuning regularizer
    lambda_anchor: float = 1.0
    # §5.17 unified objective
    lambda_loc: float = 1.0
    lambda_conf: float = 1.0
    lambda_mse: float = 1.0
    # §5.10 pooling
    pool_size: int = 7
    # architecture
    use_attention: bool = True


@dataclass
class TrainP1Config:
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 1.0e-4
    warmup_steps: int = 500
    save_ckpt_every_steps: int = 1000
    log_every_steps: int = 100
    csv_flush_every_steps: int = 100
    max_steps_per_epoch: int = 0
    num_workers: int = -1
    keep_last_step_ckpts: int = 3
    val_max_batches: int = 20
    early_stop_patience: int = 2
    ema_decay: float = 0.999


@dataclass
class TrainP2Config:
    additional_epochs: int = 4
    steps_per_epoch: int = 4000
    batch_size: int = 16
    # v3.3.0: separate LRs for denoiser and YOLO26 head
    denoiser_lr: float = 1.0e-5
    head_lr: float = 1.0e-5
    # Legacy single-lr field kept for backward compat; ignored if denoiser_lr is set
    learning_rate: float = 1.0e-5
    warmup_steps: int = 200
    log_every_steps: int = 50
    save_ckpt_every_steps: int = 500
    csv_flush_every_steps: int = 100
    val_max_batches: int = 10
    num_workers: int = -1
    early_stop_patience: int = 2
    ema_decay: float = 0.999
    # Mixture proportions (v3.3.0 naming)
    p_coco: float = 0.4
    p_apricot: float = 0.3
    p_synthetic: float = 0.3
    # Legacy aliases accepted but mapped onto p_* at load time
    coco_fraction: Optional[float] = None
    apricot_fraction: Optional[float] = None
    synthetic_fraction: Optional[float] = None

    def __post_init__(self):
        # Map legacy fraction names to canonical p_* if provided
        if self.coco_fraction is not None:
            self.p_coco = self.coco_fraction
        if self.apricot_fraction is not None:
            self.p_apricot = self.apricot_fraction
        if self.synthetic_fraction is not None:
            self.p_synthetic = self.synthetic_fraction


@dataclass
class EvalP3Config:
    alphas: list = field(default_factory=lambda: [10.0, 25.0, 50.0, 75.0, 100.0])
    score_threshold: float = 0.25
    nms_iou: float = 0.5
    on_patch_iou: float = 0.1
    yolo_raw_score_threshold: float = 0.001
    n_images: Optional[int] = None
    use_phase2_ema: bool = True
    run_full_eval: bool = True


@dataclass
class EvalNegativeControlConfig:
    n_images: int = 500
    score_threshold: float = 0.25
    use_phase2_ema: bool = True


@dataclass
class VizConfig:
    tier: str = "A"
    train_panels: bool = True
    live_dashboard: bool = True
    dashboard_port: int = 8080


@dataclass
class MultiSeedConfig:
    enabled: bool = False
    seeds: list = field(default_factory=lambda: [42])


@dataclass
class OrchestratorConfig:
    name: str
    version: str
    seed: int
    device: str
    runs_root: str
    cache_root: str
    yolo_weights: str
    data: DataConfig
    prep: PrepConfig
    dpc: DpcMathConfig
    train_p1: TrainP1Config
    train_p2: TrainP2Config
    eval_p3: EvalP3Config
    eval_negative_control: EvalNegativeControlConfig
    viz: VizConfig
    multi_seed: MultiSeedConfig

    config_path: Path = field(default=Path("."))

    @property
    def seeds(self) -> list:
        if self.multi_seed.enabled:
            return list(self.multi_seed.seeds)
        return [self.seed]

    @property
    def run_dir(self) -> Path:
        root = (self.config_path.parent / self.runs_root).resolve()
        return root / self.name

    @property
    def cache_dir(self) -> Path:
        return (self.config_path.parent / self.cache_root).resolve()

    @property
    def yolo_weights_path(self) -> Path:
        return (self.config_path.parent / self.yolo_weights).resolve()

    @property
    def coco_dir_path(self) -> Path:
        return (self.config_path.parent / self.data.coco_dir).resolve()

    @property
    def apricot_dir_path(self) -> Path:
        return (self.config_path.parent / self.data.apricot_dir).resolve()

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("config_path", None)
        return d


def load_config(path: str) -> OrchestratorConfig:
    """Load a JSON config and return a resolved OrchestratorConfig."""
    p = Path(path).resolve()
    raw = json.loads(p.read_text())

    cfg = OrchestratorConfig(
        name=raw["name"],
        version=raw.get("version", "3.3.0"),
        seed=int(raw.get("seed", 42)),
        device=raw.get("device", "auto"),
        runs_root=raw["runs_root"],
        cache_root=raw["cache_root"],
        yolo_weights=raw["yolo_weights"],
        data=DataConfig(**raw["data"]),
        prep=PrepConfig(**raw.get("prep", {})),
        dpc=DpcMathConfig(**raw.get("dpc", {})),
        train_p1=TrainP1Config(**raw.get("train_p1", {})),
        train_p2=TrainP2Config(**raw.get("train_p2", {})),
        eval_p3=EvalP3Config(**raw.get("eval_p3", {})),
        eval_negative_control=EvalNegativeControlConfig(
            **raw.get("eval_negative_control", {})
        ),
        viz=VizConfig(**raw.get("viz", {})),
        multi_seed=MultiSeedConfig(**raw.get("multi_seed", {})),
        config_path=p,
    )
    return cfg


def validate_config(cfg: OrchestratorConfig) -> list:
    """Validate paths and required files. Returns a list of issue strings."""
    issues = []
    if not cfg.yolo_weights_path.is_file():
        issues.append(f"YOLO weights missing: {cfg.yolo_weights_path}")
    if not cfg.coco_dir_path.is_dir():
        issues.append(f"COCO dir missing: {cfg.coco_dir_path}")
    if not cfg.apricot_dir_path.is_dir():
        issues.append(f"APRICOT dir missing: {cfg.apricot_dir_path}")
    if cfg.train_p1.epochs < 1:
        issues.append("train_p1.epochs must be >= 1")
    if cfg.train_p1.batch_size < 1:
        issues.append("train_p1.batch_size must be >= 1")
    if not cfg.eval_p3.alphas:
        issues.append("eval_p3.alphas must contain at least one alpha")
    # Mixture must sum to ~1
    p_sum = cfg.train_p2.p_coco + cfg.train_p2.p_apricot + cfg.train_p2.p_synthetic
    if abs(p_sum - 1.0) > 1e-3:
        issues.append(
            f"train_p2 mixture proportions sum to {p_sum:.4f}, expected 1.0 "
            f"(p_coco={cfg.train_p2.p_coco}, p_apricot={cfg.train_p2.p_apricot}, "
            f"p_synthetic={cfg.train_p2.p_synthetic})"
        )
    return issues
