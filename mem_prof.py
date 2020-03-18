

# fold
from argparse import Namespace
from transformers import *
from transformers.modeling_bart import *
from transformers import *
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

d_model =64
d_ffn = 8
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
            output_past=True,
            eos_token_ids=[2],
            pad_token_id=1,
            bos_token_id=0,
        )
hf_model = BartForConditionalGeneration(config=BartConfig(**small_kwargs))

num_parameters(hf_model) / 1e6

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
for k,v in fairseq_kwargs.items():
    if k not in args.__dict__:
        print(f'missing: {k}')
    else:
        setattr(new_args, k, v)
fs_model = BARTModel.build_model(new_args, cnn_bart_task)

assert num_parameters(hf_model) == num_parameters(fs_model)

fs_model.encoder.embed_positions, hf_model.model.encoder.embed_positions

ids = torch.tensor([[0, 6,6,6,2]]).long()
prev_output_tokens = shift_tokens_right(ids, 1)

print('***HF Forward***')
res = hf_model(ids)
#hf_log_df = hf_model.combine_logs()
hf_model.log_df

#src_lengths =
print('***Done***')

print('***Fairseq Forward***')
fs_output = fs_model(ids, None, prev_output_tokens)
#fs_log_df = fs_model.combine_logs()
print('***Done***')
