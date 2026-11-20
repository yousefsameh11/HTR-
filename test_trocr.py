import sys
sys.path.insert(0, '.')
from transformers import DeiTImageProcessor, VisionEncoderDecoderModel
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np

ckpt = 'microsoft/trocr-small-handwritten'
proc = DeiTImageProcessor.from_pretrained(ckpt)
print('DeiTImageProcessor OK')

sp_model_path = hf_hub_download(ckpt, 'sentencepiece.bpe.model')
sp = spm.SentencePieceProcessor()
sp.load(sp_model_path)
print('SentencePiece OK, vocab_size:', sp.get_piece_size())
ids = sp.encode('hello world')
print('encode test:', ids[:10])
print('decode test:', sp.decode(ids))

model = VisionEncoderDecoderModel.from_pretrained(ckpt)
print('VisionEncoderDecoderModel OK')

img = Image.fromarray((np.random.rand(384, 384, 3) * 255).astype('uint8'), 'RGB')
pv = proc(images=img, return_tensors='pt')['pixel_values']
model.eval()
with torch.no_grad():
    out = model.generate(pixel_values=pv, max_length=20)
print('generate ids:', out[0].tolist()[:20])
print('decode:', sp.decode(out[0].tolist()))
print('ALL OK')
