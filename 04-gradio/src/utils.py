
MODEL_BASE_PATH = "/cluster/home/taheeraa/code/BenchmarkTransformers/models/classification/ChestXray14"

MODELS = {
    "densenet121": {
        "experiment_name": "05-change-classifier-head",
        "folder": "densenet121_imagenet_1k_adamw_32_bce_aug_class/model.pth.tar",
        "ckpt_path": f"{MODEL_BASE_PATH}/05-change-classifier-head/densenet121_imagenet_1k_adamw_32_bce_aug_class/model.pth.tar"
    },
    "swin_in22k": {
        "experiment_name": "06-transformers-pre-trained",
        "ckpt_path": f"{MODEL_BASE_PATH}/06-transformers-pre-trained/swin_base_imagenet_1k_sgd_64_bce_aug/model.pth.tar"

    },
    "vit_in1k": {
        "experiment_name": "06-transformers-pre-trained",
        "ckpt_path": f"{MODEL_BASE_PATH}/06-transformers-pre-trained/vit_base_imagenet_1k_sgd_64_bce_True/model.pth.tar"
    },
    "swin_simim": {
        "experiment_name": "07-transformer-ssl",
        "ckpt_path": f"{MODEL_BASE_PATH}/07-transformer-ssl/swin_base_simmim/model.pth.tar"
    },
}

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax"
]
