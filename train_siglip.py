import os
import json
import logging
from datetime import datetime
import argparse

# Data processing imports
import numpy as np
import pandas as pd
from PIL import Image

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Hugging Face imports
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Progress tracking and visualization
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for model training"""
    def __init__(self):
        self.image_size = 224
        self.batch_size = 32
        self.num_epochs = 10
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.warmup_steps = 100
        self.phi_model = "microsoft/phi-2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4
        self.embedding_dim = 512
        self.temperature = 0.07
        self.save_interval = 1000  # Save checkpoint every 1000 steps
        self.max_grad_norm = 1.0   # Maximum gradient norm for clipping
        self.use_amp = True        # Use Automatic Mixed Precision
        self.loss_scale = 512.0    # Initial loss scale for AMP

class CIFAR10SigLIPDataset(Dataset):
    """Custom dataset for CIFAR10 with SigLIP training"""
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations
            transform (callable, optional): Optional transform to be applied on images
        """
        try:
            logger.info(f"Initializing dataset from {csv_file}")
            
            # Check if CSV file exists
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"CSV file not found: {csv_file}")
            
            # Load CSV data
            logger.info("Loading CSV data...")
            self.data = pd.read_csv(csv_file)
            logger.info(f"Successfully loaded {len(self.data)} samples from CSV")
            
            # Verify required columns exist
            required_columns = ['Dataset_Index'] + [f'A{i}' for i in range(1, 6)]
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {', '.join(missing_columns)}")
            
            # Load CIFAR10 dataset
            logger.info("Loading CIFAR10 dataset...")
            try:
                self.cifar10 = torchvision.datasets.CIFAR10(
                    root='./data',
                    train=False,
                    download=True
                )
                logger.info("Successfully loaded CIFAR10 dataset")
            except Exception as e:
                raise RuntimeError(f"Failed to load CIFAR10 dataset: {str(e)}")
            
            # Verify Dataset_Index values are valid
            max_idx = len(self.cifar10) - 1
            invalid_indices = self.data[
                ~self.data['Dataset_Index'].between(0, max_idx)
            ]['Dataset_Index'].unique()
            if len(invalid_indices) > 0:
                raise ValueError(
                    f"Found invalid Dataset_Index values: {invalid_indices}. "
                    f"Values must be between 0 and {max_idx}"
                )
            
            self.transform = transform
            logger.info("Dataset initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error initializing dataset: {str(e)}")
            raise

    def __len__(self):
        return len(self.data) * 5  # 5 Q&A pairs per image

    def __getitem__(self, idx):
        try:
            # Calculate image index and Q&A pair index
            image_idx = idx // 5  # Integer division to get the image index
            qa_idx = idx % 5 + 1  # Get Q&A pair number (1-5)
            
            # Get image from CIFAR10 dataset using Dataset_Index
            dataset_idx = int(self.data.iloc[image_idx]['Dataset_Index'])
            image, _ = self.cifar10[dataset_idx]
            
            # Get answer
            answer = self.data.iloc[image_idx][f'A{qa_idx}']
            
            # Clean and combine text
            answer = str(answer).strip()
            # text = f"Answer: {answer}"
            text = f"{answer}"
            
            if self.transform:
                try:
                    image = self.transform(image)
                except Exception as e:
                    logger.error(f"Error applying transform to image {image_idx}: {str(e)}")
                    raise
            
            return {
                'image': image,
                'text': text,
                'index': idx,
                'image_idx': image_idx,
                'qa_idx': qa_idx
            }
        except Exception as e:
            logger.error(f"Error loading item {idx}: {str(e)}")
            raise

class SigLIPModel(nn.Module):
    """SigLIP model implementation"""
    def __init__(self, config):
        super().__init__()
        try:
            # Use float32 for model initialization
            self.dtype = torch.float32
            logger.info(f"Initializing model with dtype: {self.dtype}")
            
            # Image encoder (ResNet50 backbone)
            self.image_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.image_encoder.fc = nn.Linear(2048, config.embedding_dim)
            
            # Load Phi model for text encoding (frozen)
            self.text_config = AutoConfig.from_pretrained(config.phi_model)
            self.text_model = AutoModelForCausalLM.from_pretrained(
                config.phi_model,
                config=self.text_config,
                torch_dtype=self.dtype
            )
            
            # Freeze Phi model
            for param in self.text_model.parameters():
                param.requires_grad = False
            
            # Text projection layer
            self.text_projection = nn.Linear(
                self.text_config.hidden_size,
                config.embedding_dim
            )
            
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / config.temperature))
            
            logger.info("SigLIP model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SigLIP model: {str(e)}")
            raise

    def encode_image(self, image):
        try:
            features = self.image_encoder(image)
            return F.normalize(features, dim=-1)
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            raise

    def encode_text(self, text_tokens):
        try:
            with torch.no_grad():
                text_features = self.text_model(
                    text_tokens,
                    output_hidden_states=True
                ).hidden_states[-1][:, -1, :]  # Use last hidden state of last token
            
            text_features = self.text_projection(text_features)
            return F.normalize(text_features, dim=-1)
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise

    def forward(self, image, text_tokens):
        try:
            image_features = self.encode_image(image)
            text_features = self.encode_text(text_tokens)
            
            # Ensure consistent dtype for matrix multiplication
            image_features = image_features.to(self.dtype)
            text_features = text_features.to(self.dtype)
            
            # Scaled pairwise cosine similarities
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            
            return logits_per_image, logits_per_text
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise

class TrainingMetrics:
    """Class to track and visualize training metrics"""
    def __init__(self):
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
    
    def update(self, metrics_dict):
        """Update batch-level metrics"""
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
    
    def update_epoch(self, metrics_dict):
        """Update epoch-level metrics"""
        for key, value in metrics_dict.items():
            self.epoch_metrics[key].append(value)
    
    def plot_metrics(self, save_dir):
        """Plot and save training metrics"""
        try:
            # Create metrics directory
            metrics_dir = os.path.join(save_dir, 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Plot epoch metrics
            plt.figure(figsize=(15, 5))
            
            # Loss plot
            plt.subplot(1, 2, 1)
            plt.plot(self.epoch_metrics['avg_loss'], label='Average Loss')
            plt.title('Training Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            # Learning rate plot
            plt.subplot(1, 2, 2)
            plt.plot(self.epoch_metrics['learning_rate'], label='Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(metrics_dir, 'epoch_metrics.png'))
            plt.close()
            
            # Plot batch metrics
            plt.figure(figsize=(15, 10))
            
            # Batch loss
            plt.subplot(2, 2, 1)
            plt.plot(self.metrics['batch_loss'], label='Batch Loss', alpha=0.5)
            plt.title('Training Loss per Batch')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            # Gradient norm
            if 'grad_norm' in self.metrics:
                plt.subplot(2, 2, 2)
                plt.plot(self.metrics['grad_norm'], label='Gradient Norm', alpha=0.5)
                plt.title('Gradient Norm per Batch')
                plt.xlabel('Batch')
                plt.ylabel('Norm')
                plt.grid(True)
                plt.legend()
            
            # Valid batches per epoch
            plt.subplot(2, 2, 3)
            plt.bar(range(len(self.epoch_metrics['valid_batches'])), 
                   self.epoch_metrics['valid_batches'],
                   label='Valid Batches')
            plt.title('Valid Batches per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Count')
            plt.legend()
            
            # AMP scaling factor
            if 'amp_scale' in self.metrics:
                plt.subplot(2, 2, 4)
                plt.plot(self.metrics['amp_scale'], label='AMP Scale', alpha=0.5)
                plt.title('AMP Scaling Factor')
                plt.xlabel('Batch')
                plt.ylabel('Scale')
                plt.grid(True)
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(metrics_dir, 'batch_metrics.png'))
            plt.close()
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame({
                'epoch': range(1, len(self.epoch_metrics['avg_loss']) + 1),
                'avg_loss': self.epoch_metrics['avg_loss'],
                'learning_rate': self.epoch_metrics['learning_rate'],
                'valid_batches': self.epoch_metrics['valid_batches']
            })
            metrics_df.to_csv(os.path.join(metrics_dir, 'training_metrics.csv'), index=False)
            
            # Save summary statistics
            summary = {
                'final_loss': self.epoch_metrics['avg_loss'][-1],
                'best_loss': min(self.epoch_metrics['avg_loss']),
                'total_epochs': len(self.epoch_metrics['avg_loss']),
                'total_batches': len(self.metrics['batch_loss']),
                'total_valid_batches': sum(self.epoch_metrics['valid_batches']),
                'avg_valid_batches_per_epoch': np.mean(self.epoch_metrics['valid_batches']),
                'final_learning_rate': self.epoch_metrics['learning_rate'][-1]
            }
            
            with open(os.path.join(metrics_dir, 'training_summary.json'), 'w') as f:
                json.dump(summary, f, indent=4)
            
            logger.info("Training metrics saved to directory: %s", metrics_dir)
            logger.info("\nTraining Summary:")
            logger.info("-----------------")
            logger.info(f"Total Epochs: {summary['total_epochs']}")
            logger.info(f"Total Batches: {summary['total_batches']}")
            logger.info(f"Final Loss: {summary['final_loss']:.4f}")
            logger.info(f"Best Loss: {summary['best_loss']:.4f}")
            logger.info(f"Valid Batches: {summary['total_valid_batches']}")
            logger.info(f"Avg Valid Batches per Epoch: {summary['avg_valid_batches_per_epoch']:.1f}")
            logger.info(f"Final Learning Rate: {summary['final_learning_rate']:.6f}")
            
        except Exception as e:
            logger.error(f"Error plotting metrics: {str(e)}")
            raise

def worker_init_fn(worker_id):
    """Initialize worker process with a unique seed"""
    worker_seed = 42 + worker_id
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)

def train_siglip(config, dataset_path):
    """Main training function"""
    try:
        logger.info("Starting SigLIP training setup...")
        
        # Initialize metrics tracker
        metrics = TrainingMetrics()
        
        # Set default dtype and device
        torch.set_default_dtype(torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Configure AMP
        if torch.cuda.is_available():
            logger.info("CUDA available, configuring AMP...")
            amp_dtype = torch.float16
            amp_enabled = True
            # Create generator for DataLoader
            generator = torch.Generator(device=device)
            generator.manual_seed(42)
        else:
            logger.info("CUDA not available, using CPU with float32...")
            amp_dtype = torch.float32
            amp_enabled = False
            generator = torch.Generator()
            generator.manual_seed(42)
        
        # Initialize gradient scaler
        scaler = torch.amp.GradScaler(enabled=amp_enabled)
        
        # Set up transforms
        transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize dataset and dataloader
        dataset = CIFAR10SigLIPDataset(dataset_path, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            generator=generator,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if config.num_workers > 0 else False
        )
        
        # Initialize and configure tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.phi_model)
        
        # Configure padding token
        if tokenizer.pad_token is None:
            logger.info("Setting up padding token...")
            if tokenizer.eos_token is None:
                special_tokens_dict = {'pad_token': '[PAD]'}
                num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
                logger.info(f"Added {num_added_tokens} special tokens: [PAD]")
            else:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Using EOS token as padding token")
        
        logger.info(f"Padding token configured: {tokenizer.pad_token}")
        
        # Initialize model
        model = SigLIPModel(config).to(config.device)
        
        # Resize token embeddings if needed
        if tokenizer.pad_token == '[PAD]':
            logger.info("Resizing model token embeddings...")
            model.text_model.resize_token_embeddings(len(tokenizer))
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=len(dataloader) * config.num_epochs
        )
        
        # Generate unique model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"siglip_phi_cifar10_{timestamp}"
        save_dir = os.path.join("models", model_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(save_dir, "config.json"), 'w') as f:
            config_dict = {k: str(v) for k, v in vars(config).items()}
            json.dump(config_dict, f, indent=4)
        
        logger.info(f"Training model: {model_name}")
        logger.info(f"Using device: {config.device}")
        logger.info(f"Number of training samples: {len(dataset)}")
        
        # Training loop
        global_step = 0
        for epoch in range(config.num_epochs):
            model.train()
            epoch_loss = 0
            valid_batches = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
            
            for batch in progress_bar:
                try:
                    # Move images to device
                    images = batch['image'].to(config.device)
                    
                    # Tokenize text
                    text_tokens = tokenizer(
                        batch['text'],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).input_ids.to(config.device)
                    
                    # Forward pass with AMP
                    with torch.amp.autocast(enabled=amp_enabled, dtype=amp_dtype, device_type="cuda"):
                        logits_per_image, logits_per_text = model(images, text_tokens)
                        
                        # Calculate loss
                        labels = torch.arange(len(images)).to(config.device)
                        
                        # Check for invalid values
                        if torch.isnan(logits_per_image).any() or torch.isinf(logits_per_image).any():
                            logger.warning("Found NaN or Inf in logits_per_image, skipping batch")
                            continue
                        
                        if torch.isnan(logits_per_text).any() or torch.isinf(logits_per_text).any():
                            logger.warning("Found NaN or Inf in logits_per_text, skipping batch")
                            continue
                        
                        # Scale logits for numerical stability
                        logits_per_image = logits_per_image / config.temperature
                        logits_per_text = logits_per_text / config.temperature
                        
                        loss_i = F.cross_entropy(logits_per_image, labels)
                        loss_t = F.cross_entropy(logits_per_text, labels)
                        loss = (loss_i + loss_t) / 2
                    
                    # Check if loss is valid
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Invalid loss value: {loss.item()}, skipping batch")
                        continue
                    
                    # Backward pass with gradient scaling
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if amp_enabled:
                        scaler.unscale_(optimizer)
                    
                    # Calculate gradient norm for monitoring
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    
                    # Optimizer step
                    if amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    
                    # Update metrics
                    metrics.update({
                        'batch_loss': loss.item(),
                        'grad_norm': grad_norm.item(),
                        'amp_scale': scaler.get_scale() if amp_enabled else 1.0
                    })
                    
                    # Update progress
                    epoch_loss += loss.item()
                    valid_batches += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.6f}",
                        'scale': f"{scaler.get_scale():.1f}" if amp_enabled else "N/A"
                    })
                    
                    # Save checkpoint
                    global_step += 1
                    if global_step % config.save_interval == 0:
                        checkpoint_path = os.path.join(
                            save_dir,
                            f"checkpoint_{global_step}.pt"
                        )
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'scaler_state_dict': scaler.state_dict() if amp_enabled else None,
                            'loss': loss.item(),
                        }, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                except Exception as e:
                    logger.error(f"Error during training step: {str(e)}")
                    continue
            
            # Log epoch statistics
            if valid_batches > 0:
                avg_epoch_loss = epoch_loss / valid_batches
                metrics.update_epoch({
                    'avg_loss': avg_epoch_loss,
                    'learning_rate': scheduler.get_last_lr()[0],
                    'valid_batches': valid_batches
                })
                logger.info(f"Epoch {epoch + 1}/{config.num_epochs} - "
                           f"Average Loss: {avg_epoch_loss:.4f} "
                           f"(Valid batches: {valid_batches}/{len(dataloader)})")
            else:
                logger.warning(f"Epoch {epoch + 1}/{config.num_epochs} - "
                             f"No valid batches processed")
        
        # Plot and save metrics
        metrics.plot_metrics(save_dir)
        
        # Save final model
        final_model_path = os.path.join(save_dir, "final_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': vars(config)
        }, final_model_path)
        logger.info(f"Training completed. Final model saved to {final_model_path}")
        
        return model_name
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def main():
    """Main function"""
    try:
        parser = argparse.ArgumentParser(description='Train SigLIP model on CIFAR10 data')
        parser.add_argument('--input-csv', type=str, required=True,
                          help='Path to the input CSV file')
        parser.add_argument('--batch-size', type=int, default=32,
                          help='Batch size for training')
        parser.add_argument('--num-epochs', type=int, default=10,
                          help='Number of training epochs')
        parser.add_argument('--learning-rate', type=float, default=1e-4,
                          help='Learning rate')
        args = parser.parse_args()
        
        # Initialize configuration
        config = Config()
        config.batch_size = args.batch_size
        config.num_epochs = args.num_epochs
        config.learning_rate = args.learning_rate
        
        # Train model
        model_name = train_siglip(config, args.input_csv)
        logger.info(f"Training completed successfully. Model saved as: {model_name}")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main() 