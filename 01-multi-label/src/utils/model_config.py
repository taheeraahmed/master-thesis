import torch

def set_criterion(loss: str, class_weights: torch.Tensor = None) -> torch.nn.Module:
    if loss == 'multi_label_soft_margin':
        criterion = torch.nn.MultiLabelSoftMarginLoss()
    elif loss == 'weighted_multi_label_soft_margin':
        criterion = torch.nn.MultiLabelSoftMarginLoss(weight=class_weights)
    return criterion


class ModelConfig():
    def __init__(self, model, loss, num_epochs, batch_size, learning_rate, test_mode, experiment_name):
        self.model = model
        self.loss = loss
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.experiment_name = experiment_name
        self.test_mode = test_mode
        self.max_steps = 80000

    def __str__(self):

        table_str = (
            f"🚀 Model Configuration 🚀\n"
            f"-------------------------------------------------\n"
            f"| Attribute         | Value                     |\n"
            f"-------------------------------------------------\n"
            f"| 🧪 Experiment Name| {self.experiment_name:<25} |\n"
            f"| 📦 Model          | {self.model:<25} |\n"
            f"| 🌟 Max Steps      | {self.max_steps:<25} |\n"
            f"| 📈 Loss Function  | {self.loss:<25} |\n"
            f"| 🔄 Epochs         | {self.num_epochs:<25} |\n"
            f"| 📏 Batch Size     | {self.batch_size:<25} |\n"
            f"| 🔍 Learning Rate  | {self.learning_rate:<25.4f} |\n"
            f"| 🔬 Test Mode      | {'Enabled' if self.test_mode else 'Disabled':<25} |\n"
            f"-------------------------------------------------"
        )
        return table_str

    def __repr__(self):
        return f'model: {self.model}, loss: {self.loss}, num_epochs: {self.num_epochs}, batch_size: {self.batch_size}, learning_rate: {self.learning_rate}'

    def __eq__(self, other):
        return self.model == other.model and self.loss == other.loss and self.num_epochs == other.num_epochs and self.batch_size == other.batch_size and self.learning_rate == other.learning_rate
