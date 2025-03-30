import logging
import numpy as np

import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report


class Trainer:
    def __init__(self, model, device, epochs=3, lr=2e-5):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.lr = lr

        self.optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
        self.scheduler = None
        self.best_metrics = {"val_loss": float("inf")}

        self.logger = logging.getLogger(__name__)

    def create_scheduler(self, total_steps):
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in train_loader:
            self.optimizer.zero_grad()
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**inputs)
            loss = outputs.loss
            loss.backward()

            clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(inputs["labels"].cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        return avg_loss, accuracy

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)

                loss = outputs.loss
                total_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(inputs["labels"].cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        report = classification_report(all_labels, all_preds, output_dict=True)
        return avg_loss, accuracy, report

    def save_model(self, path):
        torch.save(
            {"model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(), "best_metrics": self.best_metrics},
            path,
        )
        self.logger.info(f"model saved in: {path}")

    def train(self, train_loader, val_loader):
        self.create_scheduler(len(train_loader) * self.epochs)

        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            self.logger.info("Training...")
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, report = self.evaluate(val_loader)

            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            self.logger.info(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            self.logger.info(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
            self.logger.info(f"Classification Report:\n{report}")

            if val_loss < self.best_metrics["val_loss"]:
                self.best_metrics = {"val_loss": val_loss, "val_acc": val_acc, "report": report}
                self.save_model(f"models/best_model_epoch_{epoch + 1}.pt")
