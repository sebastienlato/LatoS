"""Original LatoS dense decoder transformer."""

from latos.model.config import ModelConfig
from latos.model.network import LatoModel, ModelOutput, create_model

__all__ = ["LatoModel", "ModelConfig", "ModelOutput", "create_model"]
