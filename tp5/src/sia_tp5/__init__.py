"""Autoencoders del TP5 de Sistemas de Inteligencia Artificial."""

from .autoencoder import Autoencoder
from .code_dataset import (SOURCE_LENGTH, VOCABULARY, VOCABULARY_SIZE, CodeDataset,
                           belongs_to_grammar, compiles, decode_patterns,
                           encode_sources, function_parameters,
                           generate_function_sources, load_code_dataset,
                           semantic_signature)
from .data import FontDataset, load_font, row_to_bits
from .noise import bit_flip
from .optimizers import (Adam, ExponentialDecay, GradientDescent, Momentum, RMSProp,
                         optimizer_from_config)
from .training import (fit_autoencoder, fit_denoising_autoencoder,
                       fit_mixed_denoising_autoencoder,
                       fit_multilevel_denoising_autoencoder,
                       fit_variational_autoencoder,
                       reconstruction_metrics)
from .toy_data import TOY_BAR_LABELS, toy_bars
from .variational_autoencoder import VariationalAutoencoder

__all__ = [
    "Adam",
    "Autoencoder",
    "CodeDataset",
    "ExponentialDecay",
    "FontDataset",
    "GradientDescent",
    "Momentum",
    "RMSProp",
    "SOURCE_LENGTH",
    "TOY_BAR_LABELS",
    "VOCABULARY",
    "VOCABULARY_SIZE",
    "VariationalAutoencoder",
    "bit_flip",
    "belongs_to_grammar",
    "compiles",
    "decode_patterns",
    "encode_sources",
    "fit_autoencoder",
    "fit_denoising_autoencoder",
    "fit_mixed_denoising_autoencoder",
    "fit_multilevel_denoising_autoencoder",
    "fit_variational_autoencoder",
    "function_parameters",
    "generate_function_sources",
    "load_font",
    "load_code_dataset",
    "optimizer_from_config",
    "reconstruction_metrics",
    "row_to_bits",
    "semantic_signature",
    "toy_bars",
]
