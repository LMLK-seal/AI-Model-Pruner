# FILE: model_pruner_new.py

#!/usr/bin/env python3
"""
Advanced AI Model Parameter Pruning Tool
Reduces model parameters while maintaining performance quality.
Supports various pruning strategies: magnitude-based, structured, gradual, and knowledge distillation.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
### CHANGE: Added datasets for real data handling ###
from datasets import load_dataset
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import pickle
import shutil
from tqdm import tqdm
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPruner:
    """
    Advanced model pruning class with multiple strategies
    """
    
    def __init__(self, model_path: str, output_path: str, target_reduction: float = 0.75):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.target_reduction = target_reduction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.config = None
        self.model = None
        self.tokenizer = None
        self.original_params = 0
        self.pruned_params = 0
        self.original_model_for_ft = None # Store original model for fine-tuning loss
        
        self.pruning_stats = {
            'layers_pruned': [], 'parameters_removed': 0,
            'compression_ratio': 0, 'memory_saved_mb': 0
        }
        
    def load_model(self):
        """Load the model and tokenizer from the specified path"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_path, torch_dtype=torch.float32, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model.to(self.device)

            ### CHANGE: Keep a copy of the original model on CPU for fine-tuning reference ###
            self.original_model_for_ft = AutoModel.from_pretrained(self.model_path, torch_dtype=torch.float32, trust_remote_code=True)
            
            self.original_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model loaded successfully. Original parameters: {self.original_params:,}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    ### CHANGE: New method to prepare a real dataset ###
    def _prepare_dataset(self, num_samples=100, max_length=128):
        """Prepares a small, real dataset for distillation or fine-tuning."""
        logger.info("Preparing dataset from 'wikitext'...")
        # Use a small, standard dataset. 'wikitext-2-raw-v1' is a good choice.
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train').filter(lambda x: len(x['text']) > 0)
        
        # Take a small sample to keep processing fast
        sample = dataset.select(range(min(num_samples, len(dataset))))

        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

        tokenized_sample = sample.map(tokenize_function, batched=True)
        tokenized_sample.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        # Create a DataLoader
        dataloader = DataLoader(tokenized_sample, batch_size=4)
        logger.info(f"Dataset ready with {len(sample)} samples.")
        return dataloader

    def _is_prunable(self, module) -> bool:
        prunable_types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)
        return isinstance(module, prunable_types)

    def magnitude_based_pruning(self, sparsity_ratio: float = 0.5) -> None:
        logger.info(f"Performing magnitude-based pruning with {sparsity_ratio:.1%} sparsity")
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if self._is_prunable(module) and hasattr(module, 'weight'):
                parameters_to_prune.append((module, 'weight'))

        import torch.nn.utils.prune as prune
        for module, param_name in parameters_to_prune:
            prune.l1_unstructured(module, param_name, amount=sparsity_ratio)
            prune.remove(module, param_name) # Make it permanent
        
        logger.info("Magnitude-based pruning completed")

    ### CHANGE: Correctly update the parent module during structured pruning ###
    def _find_and_replace_layer(self, module_name: str, new_layer: nn.Module):
        """Finds a layer by its name and replaces it with a new layer."""
        parent = self.model
        name_parts = module_name.split('.')
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, name_parts[-1], new_layer)

    def _prune_linear_layer(self, layer: nn.Linear, keep_ratio: float) -> Optional[nn.Linear]:
        """Prunes a linear layer and returns the new, smaller layer."""
        if not isinstance(layer, nn.Linear): return None
        
        out_features = layer.out_features
        new_out_features = max(8, int(out_features * keep_ratio)) # Ensure it's a multiple of 8 for performance
        new_out_features = min(out_features, (new_out_features // 8) * 8)

        if new_out_features >= out_features: return None

        importance = torch.norm(layer.weight.data, dim=1)
        _, indices = torch.topk(importance, new_out_features, sorted=True)

        new_layer = nn.Linear(layer.in_features, new_out_features, 
                              bias=layer.bias is not None).to(self.device, dtype=layer.weight.dtype)
        
        new_layer.weight.data = layer.weight.data[indices]
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[indices]
        
        return new_layer

    def structured_pruning(self, reduction_ratio: float = 0.3) -> None:
        logger.info(f"Performing structured pruning with {reduction_ratio:.1%} reduction")
        
        layers_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and ('attention' in name or 'feed_forward' in name or 'mlp' in name):
                layers_to_prune.append(name)
        
        for name in tqdm(layers_to_prune, desc="Structured Pruning"):
            module = dict(self.model.named_modules())[name]
            new_layer = self._prune_linear_layer(module, 1 - reduction_ratio)
            if new_layer:
                self._find_and_replace_layer(name, new_layer)
                self.pruning_stats['layers_pruned'].append(name)

        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Structured pruning completed")

    ### CHANGE: Implement fine-tuning step ###
    def fine_tune_model(self, epochs: int = 1, learning_rate: float = 1e-5):
        """Fine-tunes the model for a few epochs to recover performance."""
        logger.info(f"Fine-tuning pruned model for {epochs} epoch(s)...")
        self.model.train()
        self.original_model_for_ft.to(self.device).eval() # Move reference model to GPU
        
        dataloader = self._prepare_dataset(num_samples=50, max_length=64)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            for batch in tqdm(dataloader, desc=f"Fine-tuning Epoch {epoch+1}"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Get pruned model output
                pruned_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Get original model output (as a target)
                with torch.no_grad():
                    original_output = self.original_model_for_ft(input_ids=input_ids, attention_mask=attention_mask)

                # Use MSE on hidden states as a simple fine-tuning loss
                loss = F.mse_loss(pruned_output.last_hidden_state, original_output.last_hidden_state)
                
                loss.backward()
                optimizer.step()
        
        self.original_model_for_ft.to('cpu') # Move reference model back to CPU
        torch.cuda.empty_cache()
        logger.info("Fine-tuning complete.")
    
    ### CHANGE: Integrate real data into knowledge distillation ###
    def knowledge_distillation_pruning(self, student_ratio: float = 0.5, 
                                       epochs: int = 3, learning_rate: float = 1e-4) -> None:
        logger.info(f"Performing knowledge distillation with {student_ratio:.1%} student size")
        
        if not hasattr(self.model, 'config'):
            logger.warning("Cannot perform knowledge distillation on non-HuggingFace model.")
            return

        teacher_model = self.model
        teacher_model.eval()
        
        student_config = self.model.config
        student_config.hidden_size = int(student_config.hidden_size * student_ratio)
        student_config.num_hidden_layers = max(1, int(student_config.num_hidden_layers * student_ratio))
        student_config.num_attention_heads = max(1, int(student_config.num_attention_heads * student_ratio))
        
        student_model = AutoModel.from_config(student_config).to(self.device)
        student_model.train()
        
        dataloader = self._prepare_dataset(num_samples=200, max_length=128)
        optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
        
        logger.info(f"Training student model for {epochs} epochs...")
        for epoch in range(epochs):
            for batch in tqdm(dataloader, desc=f"Distillation Epoch {epoch+1}"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    teacher_output = teacher_model(input_ids, attention_mask)
                
                student_output = student_model(input_ids, attention_mask)
                loss = F.mse_loss(student_output.last_hidden_state, teacher_output.last_hidden_state)
                loss.backward()
                optimizer.step()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        del self.model
        self.model = student_model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Knowledge distillation completed")
    
    ### CHANGE: Add the fine-tuning call ###
    def gradual_magnitude_pruning(self, initial_sparsity: float = 0.1, 
                                  final_sparsity: float = 0.7, steps: int = 10) -> None:
        logger.info(f"Performing gradual pruning from {initial_sparsity:.1%} to {final_sparsity:.1%}")
        
        current_sparsity = 0
        for step in range(steps):
            target_sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * ((step + 1) / steps)
            pruning_rate = 1 - (1 - target_sparsity) / (1 - current_sparsity)
            
            logger.info(f"Pruning step {step+1}/{steps} - Target Sparsity: {target_sparsity:.1%}")
            self.magnitude_based_pruning(sparsity_ratio=pruning_rate)
            
            # Add fine-tuning step here to recover from pruning
            self.fine_tune_model(epochs=1)
            
            current_sparsity = target_sparsity
    
    def optimize_model_size(self):
        logger.info("Starting comprehensive model optimization...")
        self.structured_pruning(reduction_ratio=0.25)
        self.gradual_magnitude_pruning(initial_sparsity=0.1, final_sparsity=0.5, steps=5)
        self.quantize_model()
        
        self.pruned_params = sum(p.numel() for p in self.model.parameters())
        reduction_ratio = 1 - (self.pruned_params / max(1, self.original_params))
        
        self.pruning_stats.update({
            'parameters_removed': self.original_params - self.pruned_params,
            'compression_ratio': reduction_ratio,
            'memory_saved_mb': (self.original_params - self.pruned_params) * 4 / (1024**2)
        })
        
        logger.info("Optimization complete!")
        logger.info(f"Parameters reduced from {self.original_params:,} to {self.pruned_params:,}")
        logger.info(f"Compression ratio: {reduction_ratio:.1%}")

    def quantize_model(self):
        """Quantize model weights to reduce memory usage"""
        logger.info("Applying model quantization (dynamic)...")
        try:
            self.model.to('cpu') # Quantization is typically done on CPU
            quantized_model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
            self.model = quantized_model.to(self.device)
            logger.info("Dynamic quantization applied successfully")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}. Keeping original model.")
            self.model.to(self.device)

    def validate_model_output(self, test_input_text: str = "Hello, this is a test.") -> Dict[str, Any]:
        logger.info("Validating pruned model output...")
        try:
            inputs = self.tokenizer(test_input_text, return_tensors="pt").to(self.device)
            self.model.eval()
            with torch.no_grad():
                output = self.model(**inputs)
            
            output_tensor = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output
            validation_results = {
                'output_generated': True,
                'output_finite': torch.isfinite(output_tensor).all().item(),
                'output_shape': list(output_tensor.shape)
            }
            logger.info(f"Model validation completed successfully: {validation_results}")
            return validation_results
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {'output_generated': False, 'error': str(e)}

    def save_pruned_model(self) -> None:
        """Save the pruned model to the output directory"""
        logger.info(f"Saving pruned model to {self.output_path}")
        self.output_path.mkdir(parents=True, exist_ok=True)
        try:
            if hasattr(self.model, 'save_pretrained') and not isinstance(self.model, torch.jit.ScriptModule):
                self.model.save_pretrained(self.output_path)
                if self.tokenizer:
                    self.tokenizer.save_pretrained(self.output_path)
            else:
                torch.save(self.model.state_dict(), self.output_path / "model_state_dict.pt")
            
            with open(self.output_path / "pruning_stats.json", 'w') as f:
                json.dump(self.pruning_stats, f, indent=2)
            logger.info("Model saved successfully!")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def generate_report(self) -> str:
        report = f"""
AI Model Pruning Report
======================
...
""" # (Report generation is the same)
        return report

# The main CLI entry point remains for backward compatibility or server-side use.
def main():
    parser = argparse.ArgumentParser(description="Advanced AI Model Pruning Tool")
    # ... (argparse code remains the same)
    args = parser.parse_args()
    # ... (main logic remains the same)

if __name__ == "__main__":
    main()