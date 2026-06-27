from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torchvision.models import resnet34

import data.awa2_loader as awa2_data_module
import data.cub_loader as cub_data_module
import data.pbc_loader as pbc_data_module
import data.pt_loader as pt_data_module


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config_path: str
    checkpoint_path: str
    aux_dir: str
    n_concepts: int
    n_classes: int
    emb_size: int
    data_module: object
    class_names: list[str]
    class_id_base: int


def _load_awa2_class_names(root_dir):
    classes_path = Path(root_dir) / "classes.txt"
    names = []
    with classes_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                names.append(parts[1])
    if len(names) != awa2_data_module.N_CLASSES:
        raise ValueError(f"AwA2 类别数异常: 读取到 {len(names)} 个类别。")
    return names


def _load_pbc_class_names():
    return ["Neutrophil", "Eosinophil", "Basophil", "Monocyte", "Lymphocyte"]


def _load_7pt_class_names():
    return [
        "basal_cell_carcinoma",
        "nevus",
        "melanoma",
        "other_benign",
        "seborrheic_keratosis",
    ]


def load_dataset_spec(dataset="CUB-200-2011", checkpoint_path=None, aux_dir=None):
    if dataset == "CUB-200-2011":
        return DatasetSpec(
            name=dataset,
            config_path="configs/CUB-200-2011.yaml",
            checkpoint_path=checkpoint_path or "checkpoints/CUB-200-2011_22-56/SemiSupervisedConceptEmbeddingModel.pt",
            aux_dir=aux_dir or "data/D-CGFS_Auxiliary",
            n_concepts=112,
            n_classes=200,
            emb_size=32,
            data_module=cub_data_module,
            class_names=list(cub_data_module.CLASS_NAMES),
            class_id_base=1,
        )

    if dataset == "AwA2":
        config_path = "configs/AwA2.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        root_dir = config["dataset_config"]["root_dir"]
        return DatasetSpec(
            name=dataset,
            config_path=config_path,
            checkpoint_path=checkpoint_path or "checkpoints/AwA2_16-15/SemiSupervisedConceptEmbeddingModel.pt",
            aux_dir=aux_dir or "data/D-CGFS_Auxiliary_AwA2",
            n_concepts=awa2_data_module.N_CONCEPTS,
            n_classes=awa2_data_module.N_CLASSES,
            emb_size=16,
            data_module=awa2_data_module,
            class_names=_load_awa2_class_names(root_dir),
            class_id_base=0,
        )

    if dataset == "PBC":
        return DatasetSpec(
            name=dataset,
            config_path="configs/PBC.yaml",
            checkpoint_path=checkpoint_path or "checkpoints/PBC_17-01/SemiSupervisedConceptEmbeddingModel.pt",
            aux_dir=aux_dir or "data/D-CGFS_Auxiliary_PBC",
            n_concepts=pbc_data_module.N_CONCEPTS,
            n_classes=pbc_data_module.N_CLASSES,
            emb_size=32,
            data_module=pbc_data_module,
            class_names=_load_pbc_class_names(),
            class_id_base=0,
        )

    if dataset == "7pt":
        return DatasetSpec(
            name=dataset,
            config_path="configs/7pt.yaml",
            checkpoint_path=checkpoint_path or "checkpoints/7pt_17-44/SemiSupervisedConceptEmbeddingModel.pt",
            aux_dir=aux_dir or "data/D-CGFS_Auxiliary_7pt",
            n_concepts=pt_data_module.N_CONCEPTS,
            n_classes=pt_data_module.N_CLASSES,
            emb_size=32,
            data_module=pt_data_module,
            class_names=_load_7pt_class_names(),
            class_id_base=0,
        )

    raise ValueError(f"不支持的数据集: {dataset}")


def load_dataset_config(spec):
    with open(spec.config_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def build_sscbm(spec, device, checkpoint_path=None):
    from models.sscbm import SSCBM
    from train.utils import wrap_pretrained_model

    model = SSCBM(
        n_concepts=spec.n_concepts,
        n_tasks=spec.n_classes,
        emb_size=spec.emb_size,
        c_extractor_arch=wrap_pretrained_model(resnet34),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path or spec.checkpoint_path, map_location=device), strict=False)
    model.eval()
    return model
