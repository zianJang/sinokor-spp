from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AlgorithmConfig:
    clip_range: float = 0.2
    demand_lambda: float = 1.0
    entropy_lambda: float = 0.010071090711145904
    feasibility_lambda: float = 0.2828168389831236
    gae_lambda: float = 0.95
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    # TODO: Why float?
    mini_batch_size: float = 0.5
    primal_dual: bool = False
    ppo_epochs: int = 5
    stabililty_lambda: float = 1.0
    tau: float = 0.005
    type: str = "sac"
    vf_lambda: float = 0.5


@dataclass
class EnvConfig:
    CI_target: float = 1.25
    LCG_target: float = 0.95
    TEU: int = 1000
    VCG_target: float = 1.05
    bays: int = 10
    blocks: int = 2
    capacity: tuple[float] = (50.0,)
    cargo_classes: int = 6
    customer_classes: int = 2
    cv_demand: float = 0.5
    decks: int = 2
    demand_uncertainty: bool = True
    env_name: str = "mpp"
    episode_order: str = "standard"
    generalization: bool = False
    hatch_overstowage_costs: float = 0.333333
    hatch_overstowage_mask: bool = False
    block_stowage_mask: bool = False
    iid_demand: bool = True
    limit_revenue: bool = True
    long_crane_costs: float = 0.5
    normalize_obs: bool = True
    perturbation: float = 0.2
    ports: int = 4
    seed: int = 42
    spot_percentage: float = 0.3
    stabililty_difference: float = 0.1
    utilization_rate_initial_demand: float = 1.1
    weight_classes: int = 3


@dataclass
class ModelConfig:
    batch_size: int = 64
    critic_temperature: float = 1.0
    decoder_type: str = "attention"
    dropout_rate: float = 0.008972135903337364
    dyn_embed: str = "self_attention"
    embed_dim: int = 128
    encoder_type: str = "attention"
    hidden_dim: int = 512
    init_dim: int = 8
    logger: str = "wandb"
    lr_end_factor: float = 0.5
    normalization: str = "layer"
    num_decoder_layers: int = 4
    num_encoder_layers: int = 3
    num_heads: int = 8
    phase: str = "train"
    scale_max: float = 1.931286785557626
    tanh_clipping: int = 0
    tanh_squashing: bool = False
    temperature: float = 0.11243639449117128


@dataclass
class TestingConfig:
    feasibility_recovery: bool = False
    folder: str = "sac-pd"
    num_episodes: int = 30
    path: Path = Path("results/trained_models/navigating_uncertainty")


@dataclass
class ProjectionConfig:
    alpha: float = 0.01
    delta: float = 0.01
    max_iter: int = 300
    n_action: int = 20
    n_constraints: int = 25
    scale: float = 0.00025463040788043916
    slack_penalty: int = 10000
    use_early_stopping: bool = True


@dataclass
class TrainingConfig:
    lr: float = 0.00014690714579803494
    pd_lr: float = 0.000034690714579803494
    optimizer: str = "Adam"
    projection_kwargs: ProjectionConfig = field(default_factory=ProjectionConfig)
    projection_type: str = "None"
    test_data_size: int = 5000
    train_data_size: int = 7200000
    val_data_size: int = 5000
    validation_freq: float = 0.2
    validation_patience: int = 2


@dataclass
class Config:
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb_version: int = 1
