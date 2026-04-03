class PipelineRunner:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pass

    def run(self):
        # 1. Data
        train_loader, val_loader, class_names = get_dataloaders("data/raw/")
        # 2. Model
        # 3. Training
        # 4. Evaluation
        pass

