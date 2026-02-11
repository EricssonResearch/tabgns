import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from typing import cast, Dict, Any

from tabgns.utils import set_seed
from tabgns.mlp import MLP
from tabgns.trainer import TabGNSTrainer
from load_data import load_higgs, split_train_valid_test, load_covertype

@hydra.main(version_base=None, config_path="conf", config_name="simple")
def main(cfg: DictConfig) -> None:
    config_dict = cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    
    set_seed(config_dict['general']['seed'])
    
    assert config_dict['general']['problem_type'] == 'classification', \
        "This evaluation script is designed for classification problems only"
    
    if config_dict['general']['dataset'] == 'higgs':
      X, y = load_higgs()
    elif config_dict['general']['dataset'] == 'covertype':
      X, y = load_covertype()
    elif config_dict['general']['dataset'] == 'higgs_large':
      X, y = load_higgs(small=False)
    else:
      raise ValueError(f"Dataset {config_dict['general']['dataset']} not supported")
    
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split_train_valid_test(X, y, dl=False, seed=config_dict['general']['seed'])
    
    config_dict['general']['inp_dim'] = X.shape[1]
    config_dict['general']['out_dim'] = 2 if config_dict['general']['dataset'] == 'higgs' else 7
    config_dict['layer_widths'] = config_dict['search_space']['layer_widths']
    
    mlp = MLP(config_dict)
    mlp.fit(X_train, y_train, X_valid, y_valid)
    y_pred = mlp.predict(torch.tensor(X_test, dtype=torch.float32))
    
    # Convert y_test to class indices if it's one-hot encoded
    y_test_tensor = torch.tensor(y_test)
    if y_test_tensor.ndim > 1 and y_test_tensor.size(-1) > 1:
      y_test_indices = torch.argmax(y_test_tensor, dim=1)
    else:
      y_test_indices = y_test_tensor.squeeze() if y_test_tensor.ndim > 1 else y_test_tensor
      y_test_indices = y_test_indices.long()
    
    print(mlp)
    mlp_acc = (y_pred == y_test_indices).float().mean().item()
    print(f"MLP Accuracy: {mlp_acc}")

    # TabGNS Training
    trainer = TabGNSTrainer(config_dict)
    mlp_nas = trainer.fit(X_train, y_train, X_valid, y_valid)
    y_pred = mlp_nas.predict(torch.tensor(X_test, dtype=torch.float32))

    print(mlp_nas)
    nas_acc = (y_pred == y_test_indices).float().mean().item()
    print(f"TabGNS Accuracy: {nas_acc}")
    
    print()
    print(f"""
    MLP:
        Architecture: {mlp.get_architecture()}
        Params: {sum([a * b for a, b in zip([config_dict['general']['inp_dim']] + mlp.get_architecture(), mlp.get_architecture() + [config_dict['general']['out_dim']])]):,}
        Accuracy: {mlp_acc}
    TabGNS:
        Architecture: {mlp_nas.get_architecture()}
        Params: {sum([a * b for a, b in zip([config_dict['general']['inp_dim']] + mlp_nas.get_architecture(), mlp_nas.get_architecture() + [config_dict['general']['out_dim']])]):,}
        Accuracy: {nas_acc}
    """)

if __name__ == "__main__":
    main()