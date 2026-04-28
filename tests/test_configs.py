import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


def test_model_latent_dim(cfg_train: DictConfig) -> None:
    """Verify the model instantiates with emb_dim=128 and free_bits=1.0."""
    from hydra.utils import instantiate

    model_cfg = cfg_train.model
    model = instantiate(model_cfg)

    # emb_dim drives the bottleneck output size
    assert model.graph_ae.bottle_neck_encoder.d_out == 128, (
        f"Expected emb_dim=128, got {model.graph_ae.bottle_neck_encoder.d_out}"
    )
    # free_bits should be 1.0, not 0
    assert model.critic.kld_loss.free_bits == 1.0, (
        f"Expected free_bits=1.0, got {model.critic.kld_loss.free_bits}"
    )
