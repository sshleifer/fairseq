

# fold
from argparse import Namespace
from transformers import *
from transformers.modeling_bart import *
from transformers import *
import gc
import fairseq
from fairseq.models.bart import BARTModel
import torch
import os
from durbango.torch_utils import *
from durbango import *
#sd = torch.load('bart_xsum/model.pt', map_location='cpu')
args = pickle_load(os.path.expanduser('~/transformers_fork/fairseq_bart_args.pkl'))
cnn_bart_task = pickle_load(os.path.expanduser('~/transformers_fork/fairseq_cnn_bart_task.pkl'))
#args = sd['args']
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
d_model =128
d_ffn = 32
small_kwargs = dict(
            vocab_size=50264,
            d_model=d_model,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=d_ffn,
            decoder_ffn_dim=d_ffn,
            max_position_embeddings=d_model * 2,
            output_past=False,
            eos_token_ids=[2],
            pad_token_id=1,
            bos_token_id=0,
        )
#from durbango.torch_utils import
hf_model = BartForConditionalGeneration(config=BartConfig(**small_kwargs)).to(torch_device)
new_args = Namespace(**args.__dict__)

def update_args(args, small_kwargs):
    for k,v in small_kwargs.items():
        if k not in args.__dict__:
            print(f'missing: {k}')
        else:
            print(f'match: {k}')
            setattr(new_args, k, v)


fairseq_kwargs = {
    "max_source_positions": d_model*2,
    "max_target_positions": d_model*2,
    "max_tokens": d_model*2,
    "encoder_embed_dim": d_model,
    "encoder_ffn_embed_dim": d_ffn,
    "encoder_layers": 2,
    "encoder_attention_heads": 2,
    "decoder_embed_dim": d_model,
    "decoder_ffn_embed_dim": d_ffn,
    "decoder_layers": 2,
    "decoder_attention_heads": 2,
    "decoder_output_dim": d_model,
    "decoder_input_dim": d_model,
}
update_args(new_args, fairseq_kwargs)
# for k,v in fairseq_kwargs.items():
#     if k not in args.__dict__:
#         print(f'missing: {k}')
#     else:
#         setattr(new_args, k, v)
fs_model = BARTModel.build_model(new_args, cnn_bart_task).to(torch_device)

hf_model.reset_logs()
fs_model.reset_logs()
import time

assert num_parameters(hf_model) == num_parameters(fs_model)

ids = torch.tensor([[0, 6,6,6,2]]).long().to(torch_device)
prev_output_tokens = shift_tokens_right(ids, 1).to(torch_device)
nruns = 1
hf_summaries, fs_summaries = [], []


print('***Fairseq Forward***')
for _ in range(nruns):
    fs_output = fs_model(ids, None, prev_output_tokens)
    fs_summaries.append(fs_model.summary)

    fs_model.reset_logs()


print(pd.DataFrame(fs_summaries))

print('***Done***')


print('***HF Forward***')
for _ in range(nruns):
    res = hf_model(ids, decoder_input_ids=prev_output_tokens)
    hf_summaries.append(hf_model.summary)
    hf_model.reset_logs()

print(pd.DataFrame(hf_summaries))
#src_lengths =
print('***Done***')
