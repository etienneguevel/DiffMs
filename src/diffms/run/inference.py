import logging
import pickle
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

from diffms import ROOT
from diffms.analysis.visualization import MolecularVisualization
from diffms.datasets import spec2mol_dataset
from diffms.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from diffms.diffusion.extra_features_molecular import ExtraMolecularFeatures
from diffms.diffusion_model_spec2mol import Spec2MolDenoisingDiffusion
from diffms.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from diffms.spec2mol_main import load_weights


cfg_dir = str(ROOT / "configs")


def move_batch_to_device(batch, device):
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    return batch

def move_data_to_device(data, device):
    for attr in vars(data)["_store"]:
        if isinstance(x := getattr(data, attr), torch.Tensor):
            setattr(data, attr, x.to(device))
    
    return data

@hydra.main(version_base="1.3", config_path=cfg_dir, config_name="config")
def main(cfg: DictConfig):
    RDLogger.DisableLog('rdApp.warning')

    dataset_config = cfg["dataset"]
    # TODO: modify the dataset in order to take any mols
    datamodule = spec2mol_dataset.Spec2MolDataModule(cfg)
    dataset_infos = spec2mol_dataset.Spec2MolDatasetInfos(datamodule, cfg)
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

    if cfg.model.extra_features:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features, domain_features=domain_features)

    logging.info("Dataset infos:", dataset_infos.output_dims)

    # Get the other kwargs
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    model_kwargs = {
        'dataset_infos': dataset_infos,
        'train_metrics': train_metrics,
        'visualization_tools': visualization_tools,
        'extra_features': extra_features, 
        'domain_features': domain_features,
    }

    # Init the model
    model = Spec2MolDenoisingDiffusion(cfg=cfg, **model_kwargs)
    weight_path = ROOT / cfg.general.load_weights
    if not os.path.isfile:
        raise ValueError(f"The path indicated does not exist {weight_path}")

    logging.info(f"Loading weights from {weight_path}")
    model = load_weights(model, weight_path)

    # Init the dataloader
    dataloader = datamodule.test_dataloader()

    # Start itering over the batches
    device = cfg.inference.device
    num_samples = cfg.inference.num_samples

    true_mols = []
    pred_mols = []

    model.to(device)

    for i, b in enumerate(dataloader):
        logging.info(f"Treating batch {i}")
        b = move_batch_to_device(b, device)
        
        with torch.no_grad():
            output, aux = model.encoder(b)
        
        # Do the data modification
        data = b["graph"]
        if model.merge == 'mist_fp':
            data.y = aux["int_preds"][-1]
        if model.merge == 'merge-encoder_output-linear':
            encoder_output = aux['h0']
            data.y = model.merge_function(encoder_output)
        elif model.merge == 'merge-encoder_output-mlp':
            encoder_output = aux['h0']
            data.y = model.merge_function(encoder_output)
        elif model.merge == 'downproject_4096':
            data.y = model.merge_function(output)
        
        data = move_data_to_device(data, device)

        # Init the lists
        tmols = [
            Chem.inchi.MolFromInchi(data.get_example(idx).inchi)
            for idx in range(len(data))
        ]
        pmols = [list() for _ in range(len(data))]
        
        # Generate the molecules
        for _ in tqdm(range(cfg.inference.num_gen), desc="Generating predictions"):
            for idx, mol in enumerate(model.sample_batch(data)):
                pmols[idx].append(mol)

        # Extend the existing datasets
        true_mols.extend(tmols)
        pred_mols.extend(pmols)

        # Write the files
        result_dir = ROOT / cfg.inference.save_dir
        os.makedirs(result_dir, exist_ok=True)
        
        true_mol_path = result_dir / "true.pkl"
        with open(true_mol_path, "wb") as f:
            print(f"saving the true mols at {true_mol_path}")
            pickle.dump(true_mols, f)
        
        pred_mol_path = result_dir / "pred.pkl"
        with open(pred_mol_path, "wb") as f:
            print(f"saving the predicted mols at {result_dir /  'pred.pkl'}")
            pickle.dump(pred_mols, f)

        # Break when the limit is hitten
        if len(true_mols) > num_samples:
            break
    
        # Clean the temp variables
        del output, aux, data, b
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
