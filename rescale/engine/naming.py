def fold_output_name(cfg, fold):
    return f"{cfg.get('output', {}).get('experiment_name', 'ReScale')}_fold{int(fold)}"
