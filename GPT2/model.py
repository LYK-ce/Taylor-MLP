"""
Presented by KeJi
Date: 2026-05-15
GPT-2 Taylor-MLP — Model Wrapper.

Wraps HuggingFace GPT-2 with forward hooks to capture FFN inputs/outputs,
and provides replace/restore for Taylor-approximated FFN layers.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import config


# ─────────────────────────────────────────────────────────────
# GPT-2 Wrapper
# ─────────────────────────────────────────────────────────────

class GPT2Wrapper:
    """
    Wraps a HuggingFace GPT2LMHeadModel with FFN I/O hooks and Taylor replacement.

    Usage:
        wrapper = GPT2Wrapper()
        wrapper.load()

        # Collect FFN I/O
        wrapper.clear_ffn_io()
        outputs = wrapper.forward(input_ids)
        ffn_io = wrapper.get_ffn_io()  # dict: layer_i -> {"input": tensor, "output": tensor}

        # Replace layer 5 with Taylor
        cache = load_cache("Cache/GPT2/layer_5/")
        wrapper.replace_ffn(5, cache)
        wrapper.restore_ffn(5)
    """

    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or config.MODEL_NAME
        self.device = device or config.DEVICE
        self.model = None
        self.tokenizer = None
        self._ffn_io = {}          # layer_idx -> {"input": [], "output": []}
        self._hooks = []            # registered hook handles
        self._originals = {}        # layer_idx -> original mlp module (for restore)
        self._num_layers = 0

    def load(self):
        """Load GPT-2 model and tokenizer."""
        print(f"[Model] Loading {self.model_name}...")
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine number of layers
        self._num_layers = len(self.model.transformer.h)
        print(f"[Model] Loaded. Layers: {self._num_layers}, "
              f"d_model: {self.model.config.n_embd}")

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def d_model(self):
        return self.model.config.n_embd

    # ── Forward ─────────────────────────────────────────

    def forward(self, input_ids, attention_mask=None):
        """Run GPT-2 forward pass. Hooks capture FFN I/O automatically."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        return outputs

    # ── FFN Hook Management ─────────────────────────────

    def _make_ffn_hook(self, layer_idx):
        """Create a forward hook that captures FFN input/output for a layer."""

        def hook(module, input_tensor, output_tensor):
            # input_tensor is a tuple of tensors
            inp = input_tensor[0].detach().cpu()
            out = output_tensor.detach().cpu()
            if layer_idx not in self._ffn_io:
                self._ffn_io[layer_idx] = {"input": [], "output": []}
            self._ffn_io[layer_idx]["input"].append(inp)
            self._ffn_io[layer_idx]["output"].append(out)

        return hook

    def register_ffn_hooks(self):
        """Register forward hooks on all MLP submodules."""
        self._hooks = []
        for i in range(self._num_layers):
            mlp = self.model.transformer.h[i].mlp
            handle = mlp.register_forward_hook(self._make_ffn_hook(i))
            self._hooks.append(handle)

    def remove_ffn_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def clear_ffn_io(self):
        """Clear collected FFN I/O buffers."""
        self._ffn_io = {}

    def get_ffn_io(self):
        """
        Return concatenated FFN I/O tensors.
        Returns: dict layer_idx -> {"input": Tensor (N, d_model), "output": Tensor (N, d_model)}
        """
        result = {}
        for layer_idx, data in self._ffn_io.items():
            result[layer_idx] = {
                "input": torch.cat(data["input"], dim=0),
                "output": torch.cat(data["output"], dim=0),
            }
        return result

    # ── Taylor Replacement ──────────────────────────────

    def replace_ffn(self, layer_idx, cache):
        """
        Replace the FFN at layer_idx with a Taylor-approximated forward.

        cache: dict with {"centers": Tensor (k, d_model),
                          "f_values": Tensor (k, d_model),
                          "jacobians": Tensor (k, d_model, d_model)}
        """
        if layer_idx in self._originals:
            # Already replaced; restore first to avoid stacking
            self.restore_ffn(layer_idx)

        mlp = self.model.transformer.h[layer_idx].mlp
        self._originals[layer_idx] = mlp

        # Create Taylor module
        taylor_module = _TaylorFFN(cache, self.device)
        self.model.transformer.h[layer_idx].mlp = taylor_module

    def restore_ffn(self, layer_idx):
        """Restore the original FFN at layer_idx."""
        if layer_idx in self._originals:
            self.model.transformer.h[layer_idx].mlp = self._originals.pop(layer_idx)

    def restore_all_ffns(self):
        """Restore all replaced FFNs."""
        for layer_idx in list(self._originals.keys()):
            self.restore_ffn(layer_idx)


# ─────────────────────────────────────────────────────────────
# Taylor FFN Module
# ─────────────────────────────────────────────────────────────

class _TaylorFFN(nn.Module):
    """
    A drop-in replacement for GPT2MLP that performs Taylor-approximated inference.

    For input x, finds nearest precomputed center X0 and computes:
        F̂(x) = F(X0) + J(X0) @ (x - X0)
    """

    def __init__(self, cache, device):
        super().__init__()
        self.device = device
        self.centers = cache["centers"].to(device)          # (k, d_model)
        self.f_values = cache["f_values"].to(device)        # (k, d_model)
        self.jacobians = cache["jacobians"].to(device)      # (k, d_model, d_model)
        self.k = self.centers.shape[0]

    def forward(self, hidden_states):
        """
        hidden_states: (batch, seq_len, d_model) or (batch, d_model)

        Returns Taylor-approximated FFN output, same shape as input.
        """
        orig_shape = hidden_states.shape

        # Flatten to (N, d_model)
        if hidden_states.dim() == 3:
            batch, seq, d = hidden_states.shape
            x = hidden_states.reshape(-1, d)
        else:
            x = hidden_states

        # Pairwise distances: (N, k)
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x @ c.T
        x_norm = (x ** 2).sum(dim=1, keepdim=True)          # (N, 1)
        c_norm = (self.centers ** 2).sum(dim=1)              # (k,)
        dists = x_norm + c_norm.unsqueeze(0) - 2 * x @ self.centers.T  # (N, k)

        nearest = torch.argmin(dists, dim=1)  # (N,)

        # Taylor: F̂(x) = F(X0) + J(X0) @ (x - X0)
        # For efficiency, batch by center assignment
        result = torch.zeros_like(x)
        for c_idx in range(self.k):
            mask = (nearest == c_idx)
            if not mask.any():
                continue
            x_sub = x[mask]                                        # (n_c, d)
            x0 = self.centers[c_idx]                               # (d,)
            f0 = self.f_values[c_idx]                              # (d,)
            J0 = self.jacobians[c_idx]                             # (d, d)
            dx = x_sub - x0.unsqueeze(0)                           # (n_c, d)
            approx = f0.unsqueeze(0) + dx @ J0.T                   # (n_c, d)
            result[mask] = approx

        return result.reshape(orig_shape)
