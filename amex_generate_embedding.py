import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
from layers.Sub_CA import SCA
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

# Get all available GPUs detected by the system
gpus = list(range(torch.cuda.device_count()))
print('available gpus:',gpus)

class MSK(nn.Module):
    def __init__(self, device="cuda", l_layer=6):
        super(MSK, self).__init__()
        self.device = device
        # Use eager implementation for custom mask support
        self.gpt2 = GPT2Model.from_pretrained("gpt2", attn_implementation="eager",
                                              output_attentions=True, output_hidden_states=True)
        
        self.gpt2.h = self.gpt2.h[:l_layer]
        for param in self.gpt2.h.parameters():
            param.requires_grad = False

        # Gradient checkpointing reduces VRAM usage to prevent OOM
        self.gpt2.gradient_checkpointing_enable() 

    def custom_forward(self, inputs_embeds, calibrated_mask):
        # When wrapped in DataParallel, 'self.gpt2' is the raw model
        module = self.gpt2
        input_shape = inputs_embeds.size()
        
        # Position IDs must stay within 0-1023 to avoid Assertion errors
        position_ids = torch.arange(0, input_shape[1], dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)

        inputs_embeds = module.wte(inputs_embeds)
        position_embeds = module.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        for block in module.h:
            outputs = block(hidden_states, attention_mask=calibrated_mask, output_attentions=True)
            hidden_states = outputs[0]
        
        hidden_states = module.ln_f(hidden_states)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states)

    def forward(self, x_ids, calibrated_mask):
        # Diagnostic: If working, you will see GPU 0 and GPU 1 prints here
        # print(f"GPU {torch.cuda.current_device()} processing {x_ids.shape[0]} sequences", flush=True)
        
        num_heads = self.gpt2.config.n_head
        # Prepare mask for multi-head attention (Batch, 1, Seq, Seq)
        calibrated_mask = calibrated_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)

        output = self.custom_forward(inputs_embeds=x_ids, calibrated_mask=calibrated_mask).last_hidden_state
        return output

class GenPromptEmb(nn.Module):
    def __init__(self, model_name="gpt2", num_nodes=223, device='cuda', d_model=768, l_layer=6, **kwargs):  
        super(GenPromptEmb, self).__init__()
        self.device, self.d_model, self.num_nodes = device, d_model, num_nodes
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        
        # Core MSK module
        self.msk_module = MSK(device=self.device, l_layer=l_layer)
        
        # Wrap the WHOLE MSK in DataParallel to split (Batch * Nodes) across GPUs
        self.gpt2 = nn.DataParallel(self.msk_module, device_ids=gpus).cuda()
        
        self.sub_ac = SCA(d_model=self.num_nodes, n_heads=1, d_ff=4*d_model, norm='LayerNorm',
                          attn_dropout=0.1, dropout=0.1, pre_norm=True, activation="gelu",
                          res_attention=True, n_layers=1, store_attn=False).to(self.device)
        
        for param in self.sub_ac.parameters():
            param.requires_grad = False

    def _generate_mask_batch(self, input_ids):
        batch_size, seq_len = input_ids.shape
        masks = torch.zeros((batch_size, seq_len, seq_len), device=self.device)
        start_marker = self.tokenizer.encode("<", add_special_tokens=False)[0]
        end_marker = self.tokenizer.encode(">", add_special_tokens=False)[0]

        for b in range(batch_size):
            ids = input_ids[b].tolist()
            ts_indices, lang_indices, capturing = [], [], False
            for idx, tid in enumerate(ids):
                if tid == self.tokenizer.pad_token_id: continue
                if tid == start_marker: capturing = True
                if capturing: ts_indices.append(idx)
                else: lang_indices.append(idx)
                if tid == end_marker: capturing = False
            for i in lang_indices:
                for j in ts_indices:
                    masks[b, i, j] = masks[b, j, i] = -100.0
        return masks

    def generate_embeddings(self, x, y, time_ref):
        batch_size = x.shape[0]
        all_gt_prompts, all_hd_prompts = [], []
        
        # 1. Optimized String Construction
        for i in range(batch_size):
            t1, t2 = str(time_ref[i][0]), str(time_ref[i][-1])
            gt_y = str(int(y[i][0])) 
            nodes_data = x[i].to(torch.int).cpu().numpy().T

            for node_timeline in nodes_data:
                vals_x = ", ".join(map(str, node_timeline))
                all_gt_prompts.append(f"From {t1} to {t2}, the values were {vals_x} every month. The value for Y label is {gt_y}")
                all_hd_prompts.append(f"From {t1} to {t2}, the values were {vals_x} every month. Forecast the value for Y label")

        # 2. Tokenization
        gt_tok = self.tokenizer(all_gt_prompts, padding=True, return_tensors="pt").to(self.device)
        hd_tok = self.tokenizer(all_hd_prompts, padding=True, return_tensors="pt").to(self.device)

        # 3. Mask Generation
        gt_masks = self._generate_mask_batch(gt_tok['input_ids'])
        hd_masks = self._generate_mask_batch(hd_tok['input_ids'])

        # 4. Forward Pass - DataParallel splits 'gt_tok' (Batch * Nodes) sequences
        # Result shape: (Batch * Nodes, Seq_Len, D_Model)
        gt_out = self.gpt2(gt_tok['input_ids'], gt_masks)
        hd_out = self.gpt2(hd_tok['input_ids'], hd_masks)

        # 5. Reshape and Cross-Attention
        # Re-organize into (Batch, Nodes, Seq_Len, D_Model) to extract last token
        gt_emb = gt_out.view(batch_size, self.num_nodes, -1, self.d_model)[:, :, -1, :].permute(0, 2, 1)
        hd_emb = hd_out.view(batch_size, self.num_nodes, -1, self.d_model)[:, :, -1, :].permute(0, 2, 1)

        sub_out = self.sub_ac(gt_emb, hd_emb, hd_emb)
        return sub_out.permute(0, 2, 1).squeeze()