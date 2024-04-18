class ModelConfig():
    def __init__(self, model_arg, loss_arg, num_epochs, batch_size, learning_rate, test_mode, experiment_name, add_transforms, optimizer_arg, scheduler_arg):
        self.model_arg = model_arg # The argumenet from train.py 
        self.loss_arg = loss_arg # The loss function argument
        self.model = None # The actual pytorch base model
        self.criterion = None
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.experiment_name = experiment_name
        self.test_mode = test_mode
        self.img_size = None
        self.max_steps = 80000
        self.num_labels = None
        self.labels = None
        self.add_transforms = add_transforms
        self.optimizer_arg = optimizer_arg
        self.scheduler_arg = scheduler_arg

    def __str__(self):
        criterion_name = self.criterion.__class__.__name__ if self.criterion else 'None'
    
        table_str = (
            f"🚀 Model Configuration 🚀\n"
            f"-------------------------------------------------\n"
            f"| Attribute         | Value                     |\n"
            f"-------------------------------------------------\n"
            f"| 🧪 Experiment Name| {self.experiment_name:<25} |\n" 
            f"| 🤓 Transforms?    | {self.add_transforms:<25} |\n"
            f"| 📦 Model          | {self.model_arg:<25} |\n"
            f"| 🌟 Max Steps      | {self.max_steps:<25} |\n"
            f"| 📈 Loss Function  | {criterion_name:<25} |\n"
            f"| 🔄 Epochs         | {self.num_epochs:<25} |\n"
            f"| 🃏 Optimizer      | {self.optimizer_arg:<25} |\n"
            f"| 🐳 Scheduler      | {self.scheduler_arg:<25} |\n"
            f"| 🍓 Batch Size     | {self.batch_size:<25} |\n"
            f"| 🔍 Learning Rate  | {self.learning_rate:<25.4f} |\n"
            f"| 🔬 Test Mode      | {'Enabled' if self.test_mode else 'Disabled':<25} |\n"
            f"-------------------------------------------------"
        )
        return table_str

    def __repr__(self):
        return f'model: {self.model}, loss: {self.loss}, num_epochs: {self.num_epochs}, batch_size: {self.batch_size}, learning_rate: {self.learning_rate}'

    def __eq__(self, other):
        return self.model == other.model and self.loss == other.loss and self.num_epochs == other.num_epochs and self.batch_size == other.batch_size and self.learning_rate == other.learning_rate
