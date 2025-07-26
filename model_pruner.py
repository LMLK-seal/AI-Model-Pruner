# FILE: model_pruner.py

#!/usr/bin/env python3
"""
Advanced AI Model Parameter Pruning Tool
Reduces model parameters while maintaining performance quality.
Supports various pruning strategies: magnitude-based, structured, gradual, and knowledge distillation.
Author: LMLK-seal
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
import argparse
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import gc
import copy
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPruner:
    """
    Advanced model pruning class with multiple strategies.
    Final version with production-grade improvements.
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
        self.original_model_for_ft = None
        self.pruning_settings = {}
        
        self.pruning_stats = {
            'layers_pruned': [], 'parameters_removed': 0,
            'compression_ratio': 0, 'memory_saved_mb': 0
        }

    def load_model(self):
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            
            self.model = AutoModel.from_pretrained(self.model_path, torch_dtype=torch.float32, trust_remote_code=True)
            self.model.to(self.device)

            self.original_model_for_ft = AutoModel.from_pretrained(self.model_path, torch_dtype=torch.float32, trust_remote_code=True)
            
            self.original_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model loaded successfully. Original parameters: {self.original_params:,}")

            model_size_gb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**3)
            if model_size_gb > 10:
                logger.warning(f"Large model detected ({model_size_gb:.1f} GB). Ensure you have sufficient VRAM and system RAM.")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def _prepare_dataset(self, num_samples=100, max_length=128):
        try:
            from datasets import load_dataset
            from torch.utils.data import DataLoader, TensorDataset

            logger.info("Attempting to load dataset from 'wikitext'...")
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', streaming=True).filter(lambda x: len(x['text']) > 50)
            sample_list = list(dataset.take(num_samples))
            
            if not sample_list: raise ValueError("Could not retrieve samples from streaming dataset.")

            def tokenize_function(examples):
                return self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

            tokenized_sample = [tokenize_function(s) for s in sample_list]
            
            input_ids = torch.tensor([s['input_ids'] for s in tokenized_sample])
            attention_mask = torch.tensor([s['attention_mask'] for s in tokenized_sample])
            
            tensor_dataset = TensorDataset(input_ids, attention_mask)
            dataloader = DataLoader(tensor_dataset, batch_size=4, shuffle=True)
            logger.info(f"Dataset ready with {num_samples} samples.")
            return dataloader
            
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to load real dataset: {e}. Falling back to dummy data.")
            from torch.utils.data import DataLoader, TensorDataset
            vocab_size = self.config.vocab_size if hasattr(self.config, 'vocab_size') else 30000
            dummy_input_ids = torch.randint(0, vocab_size, (num_samples, max_length))
            dummy_attention_mask = torch.ones_like(dummy_input_ids)
            dummy_dataset = TensorDataset(dummy_input_ids, dummy_attention_mask)
            return DataLoader(dummy_dataset, batch_size=4)

    def _is_prunable(self, module):
        # FIX 4: More generic check for linear-like layers
        prunable_types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)
        return isinstance(module, prunable_types)

    def magnitude_based_pruning(self, sparsity_ratio: float = 0.5) -> None:
        if sparsity_ratio <= 0: return
        logger.info(f"Performing magnitude-based pruning with {sparsity_ratio:.1%} sparsity")
        parameters_to_prune = []
        for module in self.model.modules():
            if self._is_prunable(module) and hasattr(module, 'weight'):
                parameters_to_prune.append((module, 'weight'))

        import torch.nn.utils.prune as prune
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity_ratio,
        )
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        logger.info("Magnitude-based pruning completed")

    def _find_and_replace_layer(self, module_name: str, new_layer: nn.Module):
        parent = self.model
        name_parts = module_name.split('.')
        for part in name_parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        final_part = name_parts[-1]
        if final_part.isdigit():
            parent[int(final_part)] = new_layer
        else:
            setattr(parent, final_part, new_layer)

    def structured_pruning(self, reduction_ratio: float = 0.3) -> None:
        logger.info(f"Performing structured pruning with {reduction_ratio:.1%} reduction")
        
        layers_to_prune = []
        # FIX 4: More generic layer finding logic
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Common names for fully connected layers in various architectures
                if any(keyword in name.lower() for keyword in ['attention', 'feed_forward', 'mlp', 'fc', 'dense', 'classifier', 'output']):
                    layers_to_prune.append(name)
        
        from tqdm import tqdm
        logger.info(f"Identified {len(layers_to_prune)} prunable linear layers.")
        for name in tqdm(layers_to_prune, desc="Structured Pruning"):
            module = dict(self.model.named_modules())[name]
            new_layer = self._prune_linear_layer(module, 1 - reduction_ratio)
            if new_layer:
                self._find_and_replace_layer(name, new_layer)
                self.pruning_stats['layers_pruned'].append(name)

        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Structured pruning completed")

    def _prune_linear_layer(self, layer: nn.Linear, keep_ratio: float) -> Optional[nn.Linear]:
        if not isinstance(layer, nn.Linear): return None
        
        out_features = layer.out_features
        new_out_features = max(8, int(out_features * keep_ratio))
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

    def fine_tune_model(self, epochs: int = 1, learning_rate: float = 1e-5):
        logger.info(f"Fine-tuning pruned model for {epochs} epoch(s)...")
        self.model.train()
        self.original_model_for_ft.to(self.device).eval()
        
        dataloader = self._prepare_dataset(num_samples=50, max_length=64)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        from tqdm import tqdm
        for epoch in range(epochs):
            for batch in tqdm(dataloader, desc=f"Fine-tuning Epoch {epoch+1}", leave=False):
                optimizer.zero_grad()
                input_ids, attention_mask = batch[0].to(self.device), batch[1].to(self.device)

                pruned_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                with torch.no_grad():
                    original_output = self.original_model_for_ft(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.mse_loss(pruned_output.last_hidden_state, original_output.last_hidden_state)
                loss.backward()
                optimizer.step()
        
        self.original_model_for_ft.to('cpu')
        torch.cuda.empty_cache()
        logger.info("Fine-tuning complete.")
    
    def knowledge_distillation_pruning(self, student_ratio: float = 0.5, 
                                       epochs: int = 3, learning_rate: float = 1e-4) -> None:
        logger.info(f"Performing knowledge distillation with {student_ratio:.1%} student size")
        
        if not hasattr(self.model, 'config'):
            logger.warning("Cannot perform knowledge distillation on non-HuggingFace model.")
            return
        
        # FIX 1: Move teacher to CPU to save VRAM
        teacher_model = self.model.to('cpu')
        teacher_model.eval()
        logger.info("Teacher model moved to CPU to conserve VRAM.")
        
        student_config = copy.deepcopy(teacher_model.config)
        student_config.hidden_size = int(student_config.hidden_size * student_ratio)
        student_config.num_hidden_layers = max(1, int(student_config.num_hidden_layers * student_ratio))
        student_config.num_attention_heads = max(1, int(student_config.num_attention_heads * student_ratio))
        
        student_model = AutoModel.from_config(student_config).to(self.device)
        student_model.train()
        
        dataloader = self._prepare_dataset(num_samples=200, max_length=128)
        optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
        
        logger.info(f"Training student model for {epochs} epochs...")
        from tqdm import tqdm
        for epoch in range(epochs):
            for batch in tqdm(dataloader, desc=f"Distillation Epoch {epoch+1}", leave=False):
                optimizer.zero_grad()
                input_ids, attention_mask = batch[0].to(self.device), batch[1].to(self.device)
                
                with torch.no_grad():
                    # Move batch to CPU for teacher, then result back to GPU for loss calc
                    teacher_output = teacher_model(input_ids.to('cpu'), attention_mask.to('cpu'))
                    teacher_hidden_state = teacher_output.last_hidden_state.to(self.device)

                student_output = student_model(input_ids, attention_mask)
                loss = F.mse_loss(student_output.last_hidden_state, teacher_hidden_state)
                loss.backward()
                optimizer.step()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        del self.model, teacher_model
        self.model = student_model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Knowledge distillation completed")
    
    def gradual_magnitude_pruning(self, initial_sparsity: float = 0.1, 
                                  final_sparsity: float = 0.7, steps: int = 10) -> None:
        logger.info(f"Performing gradual pruning from {initial_sparsity:.1%} to {final_sparsity:.1%}")
        
        current_sparsity = 0.0
        for step in range(steps):
            target_sparsity_for_step = initial_sparsity + (final_sparsity - initial_sparsity) * ((step + 1) / steps)
            
            if current_sparsity >= 0.99:
                 logger.warning("Current sparsity is already near 100%, stopping gradual pruning.")
                 break
            
            pruning_rate = (target_sparsity_for_step - current_sparsity) / (1 - current_sparsity)
            pruning_rate = max(0, min(pruning_rate, 1.0))

            logger.info(f"Step {step+1}/{steps} - Global Target: {target_sparsity_for_step:.1%}, Pruning Rate this step: {pruning_rate:.2%}")
            
            self.magnitude_based_pruning(sparsity_ratio=pruning_rate)
            self.fine_tune_model(epochs=1)
            
            # FIX 3: Calculate sparsity by counting zero-value weights
            total_params = 0
            zero_params = 0
            for param in self.model.parameters():
                total_params += param.numel()
                zero_params += torch.sum(param == 0).item()
            current_sparsity = zero_params / total_params

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
        logger.info("Applying model quantization (dynamic)...")
        try:
            # FIX 2: Add check for model architecture compatibility
            if "bert" not in self.model.config.model_type.lower():
                logger.warning(f"Model type '{self.model.config.model_type}' may not be fully compatible with dynamic quantization. Proceeding with caution.")

            if not any(p.is_floating_point() for p in self.model.parameters()):
                logger.warning("Model does not appear to have float parameters to quantize. Skipping.")
                return

            self.model.to('cpu')
            quantized_model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
            self.model = quantized_model
            logger.info("Dynamic quantization applied successfully. Model is now on CPU.")
            self.model.to(self.device)
            logger.info(f"Quantized model moved back to {self.device}.")
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
        logger.info(f"Saving pruned model to {self.output_path}")
        self.output_path.mkdir(parents=True, exist_ok=True)
        try:
            is_quantized = 'quantized' in str(type(self.model))
            
            if hasattr(self.model, 'save_pretrained') and not is_quantized:
                self.model.save_pretrained(self.output_path)
            else:
                logger.info("Saving quantized model state_dict.")
                torch.save(self.model.state_dict(), self.output_path / "quantized_model_state_dict.pt")
            
            if self.tokenizer:
                self.tokenizer.save_pretrained(self.output_path)
            
            with open(self.output_path / "pruning_stats.json", 'w') as f:
                json.dump(self.pruning_stats, f, indent=2)
            
            with open(self.output_path / "pruning_config.json", 'w') as f:
                json.dump(self.pruning_settings, f, indent=2)

            logger.info("Model saved successfully!")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def generate_report(self) -> str:
        report = f"""
AI Model Pruning Report
======================

Original Model: {self.model_path}
Output Location: {self.output_path}

Pruning Configuration:
- Strategy: {self.pruning_settings.get('strategy', 'N/A')}
- Target Reduction: {self.pruning_settings.get('reduction', 0.0):.1%}
- Timestamp: {self.pruning_settings.get('timestamp', 'N/A')}

Parameter Reduction:
- Original Parameters: {self.original_params:,}
- Pruned Parameters: {self.pruned_params:,}
- Parameters Removed: {self.pruning_stats['parameters_removed']:,}
- Compression Ratio: {self.pruning_stats['compression_ratio']:.1%}

Memory Savings:
- Memory Saved: {self.pruning_stats['memory_saved_mb']:.1f} MB
"""
        return report.strip()

def main():
    parser = argparse.ArgumentParser(description="Advanced AI Model Pruning Tool (CLI)")
    parser.add_argument("--input_path", "-i", required=True, help="Path to input model directory")
    parser.add_argument("--output_path", "-o", required=True, help="Path to save pruned model")
    parser.add_argument("--target_reduction", "-r", type=float, default=0.75, help="Target parameter reduction ratio (default: 0.75)")
    parser.add_argument("--pruning_strategy", "-s", choices=['magnitude', 'structured', 'gradual', 'comprehensive', 'distillation'], default='comprehensive', help="Pruning strategy to use")
    parser.add_argument("--validate", "-v", action='store_true', help="Validate pruned model output")
    args = parser.parse_args()
    
    pruner = ModelPruner(model_path=args.input_path, output_path=args.output_path, target_reduction=args.target_reduction)
    
    try:
        pruner.load_model()
        
        strategy = args.pruning_strategy
        reduction = args.target_reduction
        
        pruner.pruning_settings = {
            'strategy': strategy,
            'reduction': reduction,
            'timestamp': datetime.now().isoformat()
        }

        if strategy == 'magnitude':
            pruner.magnitude_based_pruning(sparsity_ratio=reduction)
        elif strategy == 'structured':
            pruner.structured_pruning(reduction_ratio=reduction)
        elif strategy == 'gradual':
            pruner.gradual_magnitude_pruning(final_sparsity=reduction)
        elif strategy == 'distillation':
            pruner.knowledge_distillation_pruning(student_ratio=1-reduction)
        else:
            pruner.optimize_model_size()
            
        if args.validate:
            pruner.validate_model_output()
        
        pruner.save_pruned_model()
        report = pruner.generate_report()
        print("\n" + report)
        
    except Exception as e:
        logger.error(f"Error during model pruning: {e}", exc_info=True)

if __name__ == "__main__":
    main()
