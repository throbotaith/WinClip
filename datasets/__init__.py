import numpy as np
from torch.utils.data import DataLoader
from loguru import logger

from .dataset import CLIPDataset
from .mvtec import load_mvtec, mvtec_classes
from .visa import load_visa, visa_classes
from .custom import load_custom, custom_classes


mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]

load_function_dict = {
    'mvtec': load_mvtec,
    'visa': load_visa,
    'custom': load_custom,
}

dataset_classes = {
    'mvtec': mvtec_classes,
    'visa': visa_classes,
    'custom': custom_classes,
}

def denormalization(x):
    x = (((x.transpose(1, 2, 0) * std_train) + mean_train) * 255.).astype(np.uint8)
    return x

def get_dataloader_from_args(phase, **kwargs):

    dataset_inst = CLIPDataset(
        load_function=load_function_dict[kwargs['dataset']],
        category=kwargs['class_name'],
        phase=phase,
        k_shot=kwargs['k_shot'],
        experiment_indx=kwargs['experiment_indx']
    )

    if len(dataset_inst) == 0:
        logger.warning(
            f"No samples found for dataset {kwargs['dataset']} class {kwargs['class_name']} ({phase})."
        )
        return None, dataset_inst

    if phase == 'train':
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=True,
                                  num_workers=0)
    else:
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=False,
                                 num_workers=0)


    debug_str = f"===> datasets: {kwargs['dataset']}, class name/len: {kwargs['class_name']}/{len(dataset_inst)}, batch size: {kwargs['batch_size']}"
    logger.info(debug_str)

    return data_loader, dataset_inst

