#!/usr/bin/env python

from omegaconf import OmegaConf
from esm.esmfold.v1.esmfold import ESMFoldConfig

def set_esmfold_config(esm_type="esm2_3B", **kwarg):
    return OmegaConf.structured(ESMFoldConfig(esm_type=esm_type, **kwarg))
