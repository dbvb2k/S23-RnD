import math
import os
import json
import logging
import sys
import time
from datetime import datetime
import argparse
from typing import Dict, List, Optional, Tuple
from importlib.metadata import version, PackageNotFoundError
import random

# Data processing
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision
import torchvision.transforms as transforms

# Hugging Face imports
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
import bitsandbytes as bnb

# Progress tracking
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter

# Check required package versions
required_packages = {
    'bitsandbytes': '>=0.41.1',
    'transformers': '>=4.36.0',
    'peft': '>=0.7.0'
}

def check_package_version(package_name: str, min_version: str) -> bool:
    try:
        installed_version = version(package_name)
        # Remove >= from version string
        required_version = min_version.lstrip('>=')
        # Simple version comparison - assumes semantic versioning
        installed_parts = [int(x) for x in installed_version.split('.')]
        required_parts = [int(x) for x in required_version.split('.')]
        return installed_parts >= required_parts
    except PackageNotFoundError:
        return False

# Verify package versions
for package, version_req in required_packages.items():
    if not check_package_version(package, version_req):
        raise ImportError(
            f"{package} {version_req} is required but not installed. "
            f"Please run: pip install {package}{version_req}"
        )
        
# Create a basic logger initially (will be replaced with session-specific logger)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging(run_name):
    """Set up logging to both console and file"""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join("logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a unique log file name for this training session
    log_file = os.path.join(logs_dir, f"{run_name}.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # File handler for the unique log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Also configure our module logger
    module_logger = logging.getLogger(__name__)
    
    # Log initial message
    module_logger.info(f"Starting new training session: {run_name}")
    module_logger.info(f"Logs will be saved to: {log_file}")
    
    return module_logger

# Checkpoint manager
class CheckpointManager:
    """Manages model checkpoints for saving and resuming training"""
    def __init__(self, output_dir: str, model_name: str = None, logger=None):
        self.output_dir = output_dir
        self.checkpoints_dir = os.path.join(output_dir, "checkpoints")
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = model_name or "model"
        
        # Create checkpoint directory
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.logger.info(f"Checkpoint directory created at: {self.checkpoints_dir}")
        
        # Initialize checkpoint info file
        self.checkpoint_info_file = os.path.join(self.output_dir, "checkpoint_info.json")
        if not os.path.exists(self.checkpoint_info_file):
            self._save_checkpoint_info({
                "last_checkpoint": None,
                "best_checkpoint": None,
                "last_step": 0,
                "best_loss": float("inf"),
                "training_history": []
            })
    
    def _save_checkpoint_info(self, info: dict):
        """Save checkpoint info to a JSON file"""
        with open(self.checkpoint_info_file, "w") as f:
            json.dump(info, f, indent=4)
    
    def _load_checkpoint_info(self) -> dict:
        """Load checkpoint info from a JSON file"""
        if os.path.exists(self.checkpoint_info_file):
            with open(self.checkpoint_info_file, "r") as f:
                return json.load(f)
        return {
            "last_checkpoint": None,
            "best_checkpoint": None,
            "last_step": 0,
            "best_loss": float("inf"),
            "training_history": []
        }
    
    def save_checkpoint(self, trainer, step: int, loss: float, is_best: bool = False):
        """Save a checkpoint at the given step"""
        try:
            # Create checkpoint name
            checkpoint_name = f"{self.model_name}_step_{step}.pt"
            checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_name)
            
            # Load current info
            info = self._load_checkpoint_info()
            
            # Save model, optimizer, and scheduler state
            checkpoint = {
                "step": step,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict() if trainer.optimizer else None,
                "scheduler_state_dict": trainer.lr_scheduler.state_dict() if trainer.lr_scheduler else None,
                "scaler_state_dict": trainer.scaler.state_dict() if hasattr(trainer, "scaler") and trainer.scaler else None,
                "trainer_state": trainer.state.serialize(),
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "torch": torch.get_rng_state()
                },
                "loss": loss,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            # Update info
            info["last_checkpoint"] = checkpoint_name
            info["last_step"] = step
            
            # Add to training history
            history_entry = {
                "step": step,
                "loss": loss,
                "checkpoint": checkpoint_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            info["training_history"].append(history_entry)
            
            # Update best checkpoint if applicable
            if is_best or loss < info["best_loss"]:
                info["best_checkpoint"] = checkpoint_name
                info["best_loss"] = loss
                self.logger.info(f"New best checkpoint at step {step} with loss {loss:.4f}")
            
            # Save updated info
            self._save_checkpoint_info(info)
            
            self.logger.info(f"Checkpoint saved at step {step} to {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def load_latest_checkpoint(self, trainer):
        """Load the latest checkpoint if available"""
        info = self._load_checkpoint_info()
        
        if info["last_checkpoint"] is None:
            self.logger.info("No checkpoint found. Starting training from scratch.")
            return 0
        
        checkpoint_path = os.path.join(self.checkpoints_dir, info["last_checkpoint"])
        return self.load_checkpoint(trainer, checkpoint_path)
    
    def load_checkpoint(self, trainer, checkpoint_path):
        """Load a specific checkpoint"""
        try:
            if not os.path.exists(checkpoint_path):
                self.logger.warning(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
                return 0
            
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            # Load checkpoint to CPU first to avoid OOM issues
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Restore model state
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            
            # Restore optimizer state if available
            if checkpoint["optimizer_state_dict"] and trainer.optimizer:
                trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Restore scheduler state if available
            if checkpoint["scheduler_state_dict"] and trainer.lr_scheduler:
                trainer.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            # Restore scaler state if available
            if checkpoint.get("scaler_state_dict") and hasattr(trainer, "scaler") and trainer.scaler:
                trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
            # Restore trainer state
            if checkpoint.get("trainer_state"):
                trainer.state = TrainingArguments.State.from_serialized(checkpoint["trainer_state"])
            
            # Restore RNG states
            if checkpoint.get("rng_state"):
                rng_state = checkpoint["rng_state"]
                random.setstate(rng_state["python"])
                np.random.set_state(rng_state["numpy"])
                if torch.cuda.is_available() and rng_state.get("cuda"):
                    torch.cuda.set_rng_state_all(rng_state["cuda"])
                torch.set_rng_state(rng_state["torch"])
            
            self.logger.info(f"Successfully loaded checkpoint from step {checkpoint['step']} with loss {checkpoint['loss']:.4f}")
            return checkpoint["step"]
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0 

class Config:
    """Configuration for VLM training"""
    def __init__(self):
        # Model configuration
        self.base_model = "microsoft/phi-2"
        self.siglip_checkpoint = None  # Will be set via command line argument
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # LoRA configuration
        self.lora_r = 8
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        self.lora_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "dense",
            "fc1",
            "fc2"
        ]
        
        # Training configuration
        self.max_length = 512
        self.batch_size = 32  # Increased batch size
        self.gradient_accumulation_steps = 8  # Increased gradient accumulation
        self.num_epochs = 3
        self.learning_rate = 4e-4  # Increased learning rate to compensate for larger batch size
        self.weight_decay = 0.01
        self.warmup_ratio = 0.03
        self.eval_steps = 100
        self.save_steps = 1000
        self.logging_steps = 50
        
        # Dataset configuration
        self.image_size = 224
        self.num_workers = 16  # Increased number of workers
        self.pin_memory = True
        self.prefetch_factor = 4  # Increased prefetch factor
        self.persistent_workers = True
        self.use_length_sampler = False  # Disabled for custom sampler
        
        # Optimization configuration
        self.use_flash_attention = True  # Enable Flash Attention 2
        self.use_bettertransformer = True
        self.torch_compile = False
        self.gradient_checkpointing = True
        
        # Memory optimization
        self.max_grad_norm = 1.0  # Add gradient clipping
        self.optim_type = "paged_adamw_32bit"  # Use paged optimizer
        self.mixed_precision = True  # Enable mixed precision training
        self.cache_dir = "cache"  # Different cache directory
        
        # New attribute in new version
        self.embedding_dim = 512  # SigLIP embedding dimension

class VLMPreprocessor:
    """Preprocessor class for VLM training"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Add model_input_names attribute required by Trainer
        self.model_input_names = ['input_ids', 'attention_mask', 'labels']
    
    def __call__(self, text):
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
    
    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id
    
    @property
    def model_max_length(self):
        return self.tokenizer.model_max_length

class VLMTrainer(Trainer):
    """Custom trainer for VLM with image encoding"""
    def __init__(self, siglip_model, processing_class, checkpoint_manager=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.siglip_model = siglip_model
        self.siglip_model.eval()
        self.processing_class = processing_class
        self.checkpoint_manager = checkpoint_manager
        
        # Initialize metrics
        self.train_metrics = {
            'loss': [],
            'learning_rate': [],
            'tokens_per_second': [],
            'samples_per_second': []
        }
        
        # Time tracking
        self.start_time = time.time()
        self.total_tokens = 0
        self.last_checkpoint_step = 0
        self.checkpoint_interval = 100  # Save checkpoint every N steps
        self.best_loss = float('inf')
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        try:
            # Extract image and process through SigLIP
            images = inputs.pop('image').to(self.args.device)
            
            # Get image embeddings from SigLIP
            with torch.no_grad():
                image_features = self.siglip_model.encode_image(images)
            
            # Create a copy of inputs with input_ids
            model_inputs = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'labels': inputs['labels'],
                'image_features': image_features
            }
            
            # Forward pass with image features
            outputs = model(**model_inputs)
            
            loss = outputs.loss
            
            # Scale loss if num_items_in_batch is provided
            if num_items_in_batch is not None:
                batch_size = inputs['attention_mask'].size(0)
                loss = loss * (batch_size / num_items_in_batch)
            
            # Update metrics using processing_class
            self.total_tokens += inputs['attention_mask'].sum().item()
            elapsed_time = time.time() - self.start_time
            tokens_per_second = self.total_tokens / elapsed_time
            
            # Log metrics
            if self.state.global_step % self.args.logging_steps == 0:
                self.train_metrics['loss'].append(loss.item())
                self.train_metrics['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                self.train_metrics['tokens_per_second'].append(tokens_per_second)
                self.train_metrics['samples_per_second'].append(
                    self.args.train_batch_size * self.state.global_step / elapsed_time
                )
                
                # Log to tensorboard
                self.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/tokens_per_second': tokens_per_second,
                    'train/samples_per_second': self.train_metrics['samples_per_second'][-1],
                    'train/epoch': self.state.epoch,
                    'train/global_step': self.state.global_step
                })
                
                # Save checkpoint if checkpoint manager is available and it's time to save
                if self.checkpoint_manager and self.state.global_step - self.last_checkpoint_step >= self.checkpoint_interval:
                    self.save_checkpoint(loss.item())
            
            # Update best loss and save best checkpoint
            if self.checkpoint_manager and loss.item() < self.best_loss:
                self.best_loss = loss.item()
                # Save as best checkpoint
                if self.state.global_step > 0 and self.state.global_step % 50 == 0:  # Don't save too often
                    self.save_checkpoint(loss.item(), is_best=True)
            
            return (loss, outputs) if return_outputs else loss
            
        except Exception as e:
            logger.error(f"Error in compute_loss: {str(e)}")
            raise
    
    def save_checkpoint(self, loss, is_best=False):
        """Save checkpoint using the checkpoint manager"""
        if not self.checkpoint_manager:
            return
        
        step = self.state.global_step
        self.checkpoint_manager.save_checkpoint(self, step, loss, is_best=is_best)
        self.last_checkpoint_step = step
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load checkpoint using the checkpoint manager"""
        if not self.checkpoint_manager:
            return 0
        
        if checkpoint_path:
            return self.checkpoint_manager.load_checkpoint(self, checkpoint_path)
        else:
            return self.checkpoint_manager.load_latest_checkpoint(self)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Save final checkpoint when training ends"""
        super().on_train_end(args, state, control, **kwargs)
        
        if self.checkpoint_manager:
            # Get the latest loss
            loss = self.state.log_history[-1].get('loss', 0.0) if self.state.log_history else 0.0
            self.save_checkpoint(loss)
            logger.info(f"Final checkpoint saved at step {self.state.global_step}")
    
    def train(self, resume_from_checkpoint=None, **kwargs):
        """Override train to support resuming from checkpoint"""
        # Resume from checkpoint if specified
        starting_step = 0
        if resume_from_checkpoint:
            if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
                # If True, load the latest checkpoint
                starting_step = self.load_checkpoint()
                logger.info(f"Resuming training from step {starting_step}")
            elif isinstance(resume_from_checkpoint, str):
                # If string, load from specified path
                starting_step = self.load_checkpoint(resume_from_checkpoint)
                logger.info(f"Resuming training from checkpoint {resume_from_checkpoint} at step {starting_step}")
        
        # Update checkpoint interval based on save_steps
        if hasattr(self.args, 'save_steps') and self.args.save_steps > 0:
            # Use smaller of save_steps or our default
            self.checkpoint_interval = min(self.args.save_steps, self.checkpoint_interval)
        
        # Set last checkpoint step to starting step
        self.last_checkpoint_step = starting_step
        
        # Call parent train
        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

class CIFAR10VLMDataset(Dataset):
    """Dataset for VLM training using CIFAR10"""
    def __init__(self, csv_file: str, processing_class, transform=None, max_length: int = 512):
        try:
            logger.info(f"Initializing VLM dataset from {csv_file}")
            
            self.data = pd.read_csv(csv_file)
            self.processing_class = processing_class
            self.max_length = max_length
            
            # Load CIFAR10
            self.cifar10 = torchvision.datasets.CIFAR10(
                root='./data',
                train=False,
                download=True
            )
            
            # Verify required columns
            required_columns = ['Dataset_Index'] +  [f'Q{i}' for i in range(1, 6)] + [f'A{i}' for i in range(1, 6)]
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")
            
            self.transform = transform
            
            # Pre-compute text lengths for efficient sampling
            self.text_lengths = []
            self.total_items = 0
            
            # Calculate total items and pre-compute lengths
            for idx in range(len(self.data)):
                for qa_idx in range(1, 6):
                    answer = self.data.iloc[idx][f'A{qa_idx}']
                    text = f"Answer: {answer}"
                    encoded = self.processing_class(text)
                    self.text_lengths.append(len(encoded['input_ids'][0]))
                    self.total_items += 1
            
            logger.info(f"Dataset initialized with {self.total_items} QA pairs (valid indices: 0 to {self.total_items - 1})")
            logger.info(f"Number of base images: {len(self.data)}")
            logger.info(f"Number of answers per image: 5")
            
        except Exception as e:
            logger.error(f"Error initializing dataset: {str(e)}")
            raise

    def __len__(self):
        return self.total_items

    def get_length(self, idx: int) -> int:
        """Return the length of the text at given index"""
        if not 0 <= idx < self.total_items:
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {self.total_items} (valid indices: 0 to {self.total_items - 1})")
        return self.text_lengths[idx]


    def __getitem__(self, idx: int) -> Dict:
            try:
                if not 0 <= idx < self.total_items:
                    raise IndexError(f"Index {idx} out of bounds")
                    
                # Calculate image index and QA index
                image_idx = idx // 5
                qa_idx = (idx % 5) + 1
                
                # Get image
                dataset_idx = int(self.data.iloc[image_idx]['Dataset_Index'])
                image, _ = self.cifar10[dataset_idx]
                
                if self.transform:
                    image = self.transform(image)
                
                # Get question and answer
                question = self.data.iloc[image_idx][f'Q{qa_idx}']
                answer = self.data.iloc[image_idx][f'A{qa_idx}']
                
                # Format input and output text
                input_text = f"Question: {question}"
                output_text = f"Answer: {answer}"
                
                # Process input text
                input_encoded = self.processing_class(input_text)
                
                # Process full text for labels
                full_text = f"{input_text} {output_text}"
                full_encoded = self.processing_class(full_text)
                
                # Create labels with -100 for input tokens
                labels = full_encoded['input_ids'].clone()
                labels[0, :len(input_encoded['input_ids'][0])] = -100  # Mask input tokens
                
                return {
                    'image': image,
                    'input_ids': full_encoded['input_ids'].squeeze(0),
                    'attention_mask': full_encoded['attention_mask'].squeeze(0),
                    'labels': labels.squeeze(0),
                    'question': question,
                    'answer': answer
                }
                
            except Exception as e:
                logger.error(f"Error loading item {idx}: {str(e)}")
                raise

class VLMDataCollator:
    """Custom data collator for VLM training"""
    def __init__(self, processing_class, max_length: int = 512):
        self.processing_class = processing_class
        self.max_length = max_length
        
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Stack images
        images = torch.stack([f['image'] for f in features])
        
        # Stack pre-computed tensors
        input_ids = torch.stack([f['input_ids'] for f in features])
        attention_mask = torch.stack([f['attention_mask'] for f in features])
        
        # Create the final batch
        batch = {
            "image": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }
        
        return batch

class SafeSequentialSampler(Sampler):
    """A sequential sampler that ensures we never go beyond dataset bounds"""
    def __init__(self, data_source):
        self.data_source = data_source
        self.total_size = len(self.data_source)

    def __iter__(self):
        return iter(range(self.total_size))

    def __len__(self):
        return self.total_size

def find_target_modules(model) -> List[str]:
    """Find all linear layer names in the model for LoRA targeting"""
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get the last part of the name (after the last dot)
            module_name = name.split('.')[-1]
            target_modules.add(module_name)
    return list(target_modules)

def setup_model_and_tokenizer(config: Config) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Setup the Phi-2 model and tokenizer with QLoRA"""
    try:
        logger.info("Setting up model and tokenizer...")
        
        # Create cache directory
        os.makedirs(config.cache_dir, exist_ok=True)
        
        # Load tokenizer with caching
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            cache_dir=config.cache_dir
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Optimized quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_has_fp16_weight=True  # Enable INT8 weights
        )
        
        # Check for Flash Attention availability
        use_flash_attention = config.use_flash_attention
        if use_flash_attention:
            try:
                import flash_attn
                logger.info("Flash Attention 2 is available and will be used")
            except ImportError:
                logger.warning("Flash Attention 2 is not available. Falling back to standard attention.")
                use_flash_attention = False
        
        # Load model with optimizations
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "cache_dir": config.cache_dir,
        }
        
        if use_flash_attention:
            model_kwargs["use_flash_attention_2"] = True
            model_kwargs["attention_dropout"] = 0.1  # Add dropout for stability
        
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            **model_kwargs
        )
        
        # Enable memory efficient optimizations
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()  # Enable input gradients for checkpointing
        
        # Enable better transformer
        if config.use_bettertransformer:
            try:
                from optimum.bettertransformer import BetterTransformer
                model = BetterTransformer.transform(model)
                logger.info("BetterTransformer optimization enabled")
            except Exception as e:
                logger.warning(f"Failed to enable BetterTransformer: {str(e)}")
        
        # Find all possible target modules
        if not config.lora_target_modules:
            config.lora_target_modules = find_target_modules(model)
            logger.info(f"Automatically found target modules: {config.lora_target_modules}")
        
        # Prepare model for k-bit training with memory optimizations
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing
        )
        
        # Optimized LoRA config
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            use_rslora=True  # Enable rank-stabilized LoRA
        )
        
        # Get PEFT model
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        logger.info("Model and tokenizer setup complete with optimizations")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error setting up model and tokenizer: {str(e)}")
        raise

def load_siglip_model(config: Config) -> nn.Module:
    """Load the trained SigLIP model"""
    try:
        logger.info(f"Loading SigLIP model from {config.siglip_checkpoint}")
        
        # Load checkpoint
        checkpoint = torch.load(config.siglip_checkpoint)
        
        # Initialize model
        from train_siglip import SigLIPModel, Config as SigLIPConfig
        siglip_config = SigLIPConfig()
        siglip_model = SigLIPModel(siglip_config)
        
        # Load state dict
        siglip_model.load_state_dict(checkpoint['model_state_dict'])
        siglip_model.to(config.device)
        siglip_model.eval()
        
        logger.info("SigLIP model loaded successfully")
        return siglip_model
    
    except Exception as e:
        logger.error(f"Error loading SigLIP model: {str(e)}")
        raise

def train_vlm(config: Config, dataset_path: str, resume_from=None):
    """Main training function"""
    try:
        # Create a unique run name for this training session
        run_name = f"phi2_vlm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set up logging for this training session
        global logger
        logger = setup_logging(run_name)
        
        # Initialize wandb with the unique run name
        wandb.init(project="phi2-vlm", name=run_name, config=vars(config))
        
        # Setup tensorboard
        writer = SummaryWriter(f"runs/{run_name}")
        
        # Create output directory
        output_dir = os.path.join("models", run_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Log detailed training configuration
        logger.info(f"Training configuration details:")
        for key, value in vars(config).items():
            logger.info(f"  {key}: {value}")
        
        # Load models
        model, tokenizer = setup_model_and_tokenizer(config)
        siglip_model = load_siglip_model(config)
        
        # Create preprocessing class
        processing_class = VLMPreprocessor(tokenizer)
        
        # Setup transform
        transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset
        train_dataset = CIFAR10VLMDataset(
            dataset_path,
            processing_class,
            transform=transform,
            max_length=config.max_length
        )
        
        # Calculate training steps and effective batch size
        effective_batch_size = config.batch_size * config.gradient_accumulation_steps
        total_size = len(train_dataset)
        steps_per_epoch = math.ceil(total_size / effective_batch_size)
        num_training_steps = steps_per_epoch * config.num_epochs
        save_steps = min(config.save_steps, num_training_steps // 10)
        
        logger.info(f"Training configuration:")
        logger.info(f"- Total dataset size: {total_size}")
        logger.info(f"- Effective batch size: {effective_batch_size}")
        logger.info(f"- Steps per epoch: {steps_per_epoch}")
        logger.info(f"- Total training steps: {num_training_steps}")
        logger.info(f"- Save steps: {save_steps}")
        
        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(output_dir, model_name="phi2_vlm", logger=logger)
        
        # Training arguments with optimizations
        training_args = TrainingArguments(
            output_dir=output_dir,
            run_name=run_name,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            logging_steps=config.logging_steps,
            save_steps=save_steps,
            save_strategy="steps",
            eval_strategy="no",
            load_best_model_at_end=False,
            report_to=["tensorboard", "wandb"],
            remove_unused_columns=False,
            fp16=True,
            bf16=False,  # Disable bfloat16 when using flash attention
            optim="paged_adamw_32bit",
            dataloader_num_workers=config.num_workers,
            dataloader_pin_memory=config.pin_memory,
            dataloader_prefetch_factor=config.prefetch_factor,
            dataloader_persistent_workers=config.persistent_workers,
            group_by_length=False,  # Disable length sampling to use our custom sampler
            save_total_limit=3,
            gradient_checkpointing=config.gradient_checkpointing,
            torch_compile=config.torch_compile,
            max_steps=num_training_steps,
            disable_tqdm=False,
            log_level="info",
            logging_first_step=True
        )
        
        # Initialize trainer with custom sampler
        trainer = VLMTrainer(
            siglip_model=siglip_model,
            processing_class=processing_class,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=VLMDataCollator(processing_class, config.max_length),
            checkpoint_manager=checkpoint_manager
        )
        
        # Override the default sampler with our safe sampler
        def get_train_dataloader():
            return DataLoader(
                train_dataset,
                batch_size=training_args.per_device_train_batch_size,
                sampler=SafeSequentialSampler(train_dataset),
                num_workers=training_args.dataloader_num_workers,
                pin_memory=training_args.dataloader_pin_memory,
                prefetch_factor=training_args.dataloader_prefetch_factor,
                persistent_workers=training_args.dataloader_persistent_workers,
                collate_fn=trainer.data_collator
            )
        
        trainer.get_train_dataloader = get_train_dataloader
        
        # Check if we're resuming training
        resume_training = False
        if resume_from:
            logger.info(f"Will attempt to resume training from: {resume_from}")
            resume_training = True
        
        # Train
        logger.info("Starting training...")
        start_time = time.time()
        train_result = trainer.train(resume_from_checkpoint=resume_training)
        training_time = time.time() - start_time
        
        # Save final model
        trainer.save_model()
        
        # Save training state
        trainer.state.save_to_json(
            os.path.join(output_dir, "trainer_state.json")
        )
        
        # Close tensorboard writer
        writer.close()
        
        # Log final metrics
        final_metrics = {
            'final_loss': trainer.state.log_history[-1].get('loss', None),
            'avg_tokens_per_second': np.mean(trainer.train_metrics['tokens_per_second']),
            'total_training_time': training_time,
            'total_steps': trainer.state.global_step,
            'training_loss': train_result.training_loss,
            'epoch': train_result.epoch
        }
        
        logger.info("\nTraining Complete!")
        logger.info(f"Final Loss: {final_metrics['final_loss']:.4f}")
        logger.info(f"Average Tokens/Second: {final_metrics['avg_tokens_per_second']:.2f}")
        logger.info(f"Total Training Time: {final_metrics['total_training_time'] / 3600:.2f} hours")
        logger.info(f"Total Steps: {final_metrics['total_steps']}")
        logger.info(f"Final Epoch: {final_metrics['epoch']:.2f}")
        
        # Save metrics
        with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
            json.dump(final_metrics, f, indent=4)
        
        # Log the path to the trained model
        logger.info(f"Trained model saved to: {output_dir}")
        
        return final_metrics
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def main():
    """Main function"""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join("logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    try:
        parser = argparse.ArgumentParser(description='Train Phi-2 as VLM using QLoRA')
        parser.add_argument('--input-csv', type=str, required=True,
                          help='Path to the input CSV file')
        parser.add_argument('--siglip-checkpoint', type=str, required=True,
                          help='Path to the trained SigLIP model checkpoint')
        parser.add_argument('--batch-size', type=int, default=4,
                          help='Batch size for training')
        parser.add_argument('--num-epochs', type=int, default=3,
                          help='Number of training epochs')
        parser.add_argument('--learning-rate', type=float, default=2e-4,
                          help='Learning rate')
        parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                          help='Gradient accumulation steps')
        parser.add_argument('--resume', action='store_true',
                          help='Resume training from the last checkpoint')
        parser.add_argument('--checkpoint-dir', type=str,
                          help='Directory containing checkpoints to resume from (looks for the latest checkpoint)')
        args = parser.parse_args()
        
        # Verify input files exist
        if not os.path.exists(args.input_csv):
            raise FileNotFoundError(f"Input CSV file not found: {args.input_csv}")
        if not os.path.exists(args.siglip_checkpoint):
            raise FileNotFoundError(f"SigLIP checkpoint file not found: {args.siglip_checkpoint}")
        
        # Check resume options
        resume_from = None
        if args.resume:
            resume_from = True
            logger.info("Will attempt to resume training from the last checkpoint")
        elif args.checkpoint_dir:
            if not os.path.exists(args.checkpoint_dir):
                raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")
            resume_from = args.checkpoint_dir
            logger.info(f"Will attempt to resume training from checkpoint directory: {args.checkpoint_dir}")
        
        # Initialize config
        config = Config()
        config.batch_size = args.batch_size
        config.num_epochs = args.num_epochs
        config.learning_rate = args.learning_rate
        config.siglip_checkpoint = args.siglip_checkpoint
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
        
        # Log input arguments
        logger.info("Starting training with arguments:")
        logger.info(f"  Input CSV: {args.input_csv}")
        logger.info(f"  SigLIP checkpoint: {args.siglip_checkpoint}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Number of epochs: {args.num_epochs}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
        logger.info(f"  Resume training: {args.resume}")
        if args.checkpoint_dir:
            logger.info(f"  Checkpoint directory: {args.checkpoint_dir}")
        
        # Train model
        metrics = train_vlm(config, args.input_csv, resume_from=resume_from)
        logger.info("Training completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        # Print the full traceback for debugging
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 

