def build_model(cfg):
    from rescale.build import build_model as build
    return build(cfg)

__all__ = ["build_model"]
