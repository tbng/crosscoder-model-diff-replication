import http.server
import json
import os
import pprint
import socketserver
import threading
import webbrowser
from typing import NamedTuple, Optional, Union

import einops
import plotly.express as px
import torch
import torch.nn.functional as F
# from tiny_dashboard.dashboard_implementations import \
#     CrosscoderOnlineFeatureDashboard
from torch import nn
from transformer_lens import HookedTransformer

from crosscoder import CrossCoder
from utils import get_gsm8k_dataset

torch.set_grad_enabled(False)  # important for memory saving


device = 'cuda:0'

base_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    device=device,
    dtype=torch.bfloat16,
)

math_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B",
    device=device,
    dtype=torch.bfloat16,
)

cross_coder = CrossCoder.load('version_2', 1)
collect_layer = 14

# dataset
eos = base_model.tokenizer.special_tokens_map['eos_token']
train_questions, train_answers = get_gsm8k_dataset(split='train')
merged_prompts = [q + f" {eos} " + a for q, a in zip(train_questions, train_answers)]
all_tokens = base_model.tokenizer(merged_prompts, padding=False, return_tensors="pt")  # for feature viz it's best to disable padding



# Check histograms similar to Anthropic's paper
norms = cross_coder.W_dec.norm(dim=-1)
norms.shape

relative_norms = norms[:, 1] / norms.sum(dim=-1)
relative_norms.shape

fig = px.histogram(
    relative_norms.detach().cpu().numpy(),
    title="Gemma 2 2B Base vs IT Model Diff",
    labels={"value": "Relative decoder norm strength"},
    nbins=200,
)

fig.update_layout(showlegend=False)
fig.update_yaxes(title_text="Number of Latents")

# Update x-axis ticks
fig.update_xaxes(
    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
    ticktext=['0', '0.25', '0.5', '0.75', '1.0']
)

fig.show()

PORT = 8888


def display_vis_inline(filename: str, height: int = 850):
    '''
    Displays the HTML files in local browser instead of Colab
    '''
    global PORT

    def serve(directory):
        os.chdir(directory)
        handler = http.server.SimpleHTTPRequestHandler

        try:
            with socketserver.TCPServer(("", PORT), handler) as httpd:
                print(f"Serving at http://localhost:{PORT}/{filename}")
                httpd.serve_forever()
        except OSError as e:
            print(f"Port {PORT} is already in use. Trying port {PORT + 1}")
            global PORT
            PORT += 1
            serve(directory)

    # Start server in background thread
    thread = threading.Thread(target=serve, args=(os.getcwd(),), daemon=True)
    thread.start()

    # Open in default browser
    webbrowser.open(f'http://localhost:{PORT}/{filename}')

    return thread


# Usage:
filename = "_feature_vis_demo.html"
sae_vis_data.save_feature_centric_vis(filename)
server_thread = display_vis_inline(filename)
