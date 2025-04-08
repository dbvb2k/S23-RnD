import os
import json
import logging
import time
from datetime import datetime
import argparse
from typing import Dict, List, Optional, Tuple
from importlib.metadata import version, PackageNotFoundError

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

# Data processing
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_vlm.log')
    ]
)
logger = logging.getLogger(__name__)

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
        self.batch_size = 8  # Increased from 4
        self.gradient_accumulation_steps = 8  # Increased from 4
        self.num_epochs = 3
        self.learning_rate = 2e-4
        self.weight_decay = 0.01
        self.warmup_ratio = 0.03
        self.eval_steps = 100
        self.save_steps = 1000  # Increased from 500
        self.logging_steps = 50  # Increased from 10
        
        # Dataset configuration
        self.image_size = 224
        self.num_workers = 8  # Increased from 4
        self.use_length_sampler = True  # Enable length-based sampling
        self.pin_memory = True  # Enable pinned memory
        self.prefetch_factor = 2  # Enable prefetching
        self.persistent_workers = True  # Keep workers alive between epochs
        
        # Optimization configuration
        self.use_flash_attention = False  # Disabled by default due to triton dependency
        self.use_bettertransformer = True  # Enable BetterTransformer
        self.torch_compile = False  # Disabled by default when using quantization
        self.gradient_checkpointing = True  # Enable gradient checkpointing

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
    def __init__(self, siglip_model, processing_class, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.siglip_model = siglip_model
        self.siglip_model.eval()
        self.processing_class = processing_class
        
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
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        try:
            # Extract image and text inputs
            images = inputs.pop('image').to(self.args.device)
            
            # Get image embeddings from SigLIP
            with torch.no_grad():
                image_features = self.siglip_model.encode_image(images)
            
            # Add image features to inputs
            inputs['image_features'] = image_features
            
            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Update metrics using processing_class
            self.total_tokens += inputs['input_ids'].ne(self.processing_class.pad_token_id).sum().item()
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
            
            return (loss, outputs) if return_outputs else loss
            
        except Exception as e:
            logger.error(f"Error in compute_loss: {str(e)}")
            raise

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
            required_columns = ['Dataset_Index'] + [f'A{i}' for i in range(1, 6)]
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
                raise IndexError(f"Index {idx} is out of bounds for dataset of size {self.total_items} (valid indices: 0 to {self.total_items - 1})")
                
            # Calculate image index and answer index
            image_idx = idx // 5
            answer_idx = (idx % 5) + 1
            
            if image_idx >= len(self.data):
                raise IndexError(f"Calculated image_idx {image_idx} is out of bounds for data of size {len(self.data)}")
            
            # Get image
            dataset_idx = int(self.data.iloc[image_idx]['Dataset_Index'])
            if not 0 <= dataset_idx < len(self.cifar10):
                raise IndexError(f"Dataset index {dataset_idx} is out of bounds for CIFAR10 dataset of size {len(self.cifar10)}")
                
            image, _ = self.cifar10[dataset_idx]
            
            if self.transform:
                image = self.transform(image)
            
            # Get answer
            answer = self.data.iloc[image_idx][f'A{answer_idx}']
            
            # Format text
            text = f"Answer: {answer}"
            
            # Process text using processing_class
            encoded = self.processing_class(text)
            
            return {
                'image': image,
                'text': text,
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'labels': encoded['input_ids'].squeeze(0)  # Add labels for training
            }
            
        except Exception as e:
            logger.error(f"Error loading item {idx}: {str(e)}")
            logger.error(f"Dataset size: {self.total_items}, Image count: {len(self.data)}, CIFAR10 size: {len(self.cifar10)}")
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
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Quantization config with optimal settings
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
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
        }
        
        if use_flash_attention:
            model_kwargs["use_flash_attention_2"] = True
        
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            **model_kwargs
        )
        
        # Enable gradient checkpointing
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Enable better transformer
        if config.use_bettertransformer:
            try:
                model = model.to_bettertransformer()
                logger.info("BetterTransformer optimization enabled")
            except Exception as e:
                logger.warning(f"Failed to enable BetterTransformer: {str(e)}")
        
        # Find all possible target modules
        if not config.lora_target_modules:
            config.lora_target_modules = find_target_modules(model)
            logger.info(f"Automatically found target modules: {config.lora_target_modules}")
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Get PEFT model
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        logger.info("Model and tokenizer setup complete")
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

def train_vlm(config: Config, dataset_path: str):
    """Main training function"""
    try:
        # Initialize wandb with a unique run name
        run_name = f"phi2_vlm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project="phi2-vlm", name=run_name, config=vars(config))
        
        # Setup tensorboard
        writer = SummaryWriter(f"runs/{run_name}")
        
        # Create output directory
        output_dir = os.path.join("models", run_name)
        os.makedirs(output_dir, exist_ok=True)
        
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
        
        # Calculate training steps
        num_training_steps = int((len(train_dataset) / config.batch_size) * config.num_epochs)
        save_steps = min(config.save_steps, num_training_steps // 10)  # Save at least 10 times during training
        
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
            group_by_length=config.use_length_sampler,
            save_total_limit=3,
            gradient_checkpointing=config.gradient_checkpointing,
            torch_compile=config.torch_compile,
            max_steps=num_training_steps,  # Add max_steps to prevent overrunning
            disable_tqdm=False,  # Enable progress bar for better monitoring
            log_level="info",
            logging_first_step=True  # Log the first training step
        )
        
        # Initialize trainer with custom data collator
        trainer = VLMTrainer(
            siglip_model=siglip_model,
            processing_class=processing_class,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=VLMDataCollator(processing_class, config.max_length)
        )
        
        # Log dataset and training configuration
        logger.info(f"Starting training with:")
        logger.info(f"- Dataset size: {len(train_dataset)}")
        logger.info(f"- Batch size: {config.batch_size}")
        logger.info(f"- Gradient accumulation steps: {config.gradient_accumulation_steps}")
        logger.info(f"- Total training steps: {num_training_steps}")
        logger.info(f"- Save steps: {save_steps}")
        
        # Train
        logger.info("Starting training...")
        train_result = trainer.train()
        
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
            'total_training_time': time.time() - trainer.start_time,
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
        
        return final_metrics
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def main():
    """Main function"""
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
        args = parser.parse_args()
        
        # Initialize config
        config = Config()
        config.batch_size = args.batch_size
        config.num_epochs = args.num_epochs
        config.learning_rate = args.learning_rate
        config.siglip_checkpoint = args.siglip_checkpoint
        
        # Train model
        metrics = train_vlm(config, args.input_csv)
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main() 