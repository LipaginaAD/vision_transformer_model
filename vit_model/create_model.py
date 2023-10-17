"""

"""
import torch
from torch import nn
from math import sqrt

def position_embeddings(max_len: int,
                        d_model: int):
  """
  Create tensor with position embeddings for vision transformer

  Args:
    max_len - Number of patches plus one for cls token
    d_model -  Number of dimentions of latent vector

  Returns:
    torch.Tensor with shape (1, max_len, d_model) with position embeddings

  Usage example:
    position_embeddings = position_embeddings(max_len=65,
                                              d_model=512)
  """
  pos_emb = torch.zeros(max_len, d_model)
  position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
  div_term = 1/(10000**(torch.arange(0, d_model, 2).float() / d_model))
  pos_emb[:, 0::2] = torch.sin(position * div_term)
  pos_emb[:, 1::2] = torch.cos(position * div_term)
  return pos_emb.unsqueeze(0)

def qkv_dot_product(q, k, v, d_model):
  """
  Takes three tensors QUERY, KEY and VALUE and returns Attention
  """
  q_k_t = torch.matmul(q, torch.transpose(k, 3, 2))
  softmax = torch.softmax(q_k_t / 1/(sqrt(d_model)), dim=-1)
  return torch.matmul(softmax, v)

class EmbeddedPatches(nn.Module):
  def __init__(self, batch_size, patch_size, max_len, d_model):
    super().__init__()

    self.batch_size = batch_size
    self.patch_size = patch_size
    self.max_len = max_len
    self.d_model = d_model


    self.patch = nn.Sequential(
        nn.Conv2d(3, (self.patch_size**2) * 3, kernel_size=self.patch_size, stride=self.patch_size), # (batch_size, color, h, w) -> (batch_size, (patch_size**2)*3, h/patch_size, w/patch_size)
        nn.Flatten(2, 3)) # (batch_size, (patch_size**2)*3, h/patch_size, w/patch_size) -> (batch_size, (patch_size**2)*3, h*w/patch_size**2)

    self.embed = nn.Linear((self.patch_size**2) * 3, self.d_model)

    self.cls_token = nn.Parameter(torch.rand(self.batch_size, 1, self.d_model), requires_grad=True)

    self.register_buffer( 'pe' , position_embeddings(max_len=self.max_len, d_model=self.d_model), persistent= False )

  def forward(self, x):
    patches = self.patch(x)
    x = torch.permute(patches, (0, 2, 1))
    patches_plus_cls = torch.cat([self.cls_token, self.embed(x)], dim=1)

    return patches_plus_cls + self.pe[:patches_plus_cls.size(0), :].repeat(self.batch_size, 1, 1)


class MSA(nn.Module):
  def __init__(self, batch_size, max_len, d_model, num_heads):
    super().__init__()

    self.batch_size = batch_size
    self.max_len = max_len
    self.d_model = d_model
    self.num_heads = num_heads

    self.head_dim = int(d_model / num_heads)
    self.qkv_weights = nn.Linear(d_model, 3*d_model)
    self.linear_msa = nn.Linear(d_model, d_model)

  def forward(self, x):
    # Get qkv tensor, result shape=(batch_size, max_len, d_model*3)
    qkv = self.qkv_weights(x)

    # Reshape tensor to separate it on heads, result shape=(batch_size, max_len, num_heads, head_dim*3)
    qkv = torch.reshape(qkv, (self.batch_size, self.max_len, self.num_heads, self.head_dim*3))

    # Permute tensor, result shape=(batch_size, num_heads, max_len, head_dim*3)
    qkv = torch.permute(qkv, (0, 2, 1, 3))

    # Separate qkv to q, k, v tensors
    q, k, v = torch.chunk(qkv, 3, dim=-1)

    # Get dot product, result_shape=(batch_size, num_heads, max_len, head_dim)
    heads = qkv_dot_product(q, k, v, self.d_model)

    # # Permute tensor, result_shape=(batch_size, max_len, num_heads, head_dim)
    heads = torch.permute(heads, (0, 2, 1, 3))

    # Reshape tensor, result_shape=(batch_size, max_len, d_model)
    msa = torch.reshape(heads, (self.batch_size, self.max_len, self.d_model))

    return self.linear_msa(msa)

class TransformerEnc(nn.Module):
  def __init__(self, batch_size, max_len, d_model, num_heads):
    super().__init__()

    self.batch_size = batch_size
    self.max_len = max_len
    self.d_model = d_model
    self.num_heads = num_heads

    self.ln1 = nn.LayerNorm(self.d_model)
    self.ln2 = nn.LayerNorm(self.d_model)
    self.dropout = nn.Dropout(0.1)
    self.dropout2 = nn.Dropout(0.1)
    self.add_module('msa', MSA(self.batch_size, self.max_len, self.d_model, self.num_heads))
    self.mlp = nn.Sequential(
        nn.Linear(self.d_model, self.d_model*2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(self.d_model*2, self.d_model)
    )

  def forward(self, x):
    normalize = self.ln1(x)
    msa = self.dropout(self.msa(normalize))
    x_1 = msa + x
    return self.dropout2(self.mlp(self.ln2(x_1))) + x_1

class ViT(nn.Module):
  def __init__(self, batch_size, patch_size, max_len, d_model, num_heads, num_classes, num_blocks):
    super().__init__()

    self.batch_size = batch_size
    self.patch_size = patch_size
    self.max_len = max_len
    self.d_model = d_model
    self.num_heads = num_heads
    self.num_classes = num_classes
    self.num_blocks = num_blocks

    self.transf_encoder = nn.ModuleList([
        TransformerEnc(self.batch_size, self.max_len, self.d_model, self.num_heads) for _ in range(self.num_blocks)])
    self.mlp_cls = nn.Sequential(
        nn.Linear(self.d_model, self.num_classes)
    )
    self.add_module('embed_patches', EmbeddedPatches(self.batch_size, self.patch_size, self.max_len, self.d_model))
    self.ln = nn.LayerNorm(self.d_model)

  def forward(self, x):
    output = self.embed_patches(x)

    # Loop through transformer encoder blocks
    for block in self.transf_encoder:
      output = block(output)

    # Take only cls token
    output = output[:, 0]

    return self.mlp_cls(self.ln(output))
