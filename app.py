
import argparse
import os
import sys
import math

import torch
from torch import nn
from torch import optim

import flask
from flask import Flask
from flask import url_for
from flask import request
from flask_cors import CORS

import numpy as np
import pandas as pd

from base64 import b64encode
from glob import glob
import json
from tqdm import tqdm

# create Flask app
app = Flask(__name__)
CORS(app)


b = 4
def predict(x, a, mu):
    return 1/(1+((a*(x-mu)).pow(b)).sum(1))



def predicate(x0, subset):
    '''subset boolean array of selection'''

    ## prepare training data
    x = torch.from_numpy(x0.astype(np.float32))
    x_mean = x.mean(0)
    x_std = x.std(0)+1
    x = (x-x_mean)/(x_std)
    label = torch.from_numpy(subset).float()

    bce = nn.BCELoss()
    a = torch.randn(x.shape[1]).requires_grad_(True)
    mu = torch.randn(x.shape[1]).requires_grad_(True)
    optimizer = optim.SGD([mu, a,], lr=1e-2, momentum=0.9, weight_decay=0.01)
    for e in range(3000):
        pred = predict(x, a, mu)
        l = bce(pred, label)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if e % 500 == 0:
            print('loss', l.item())
    a.detach_()
    mu.detach_()

#     plt.stem(a.abs().numpy())
#     plt.show()

    r = 1/a.abs()
    print(
        'accuracy',
        ((pred>0.5).float() == label).float().sum().item(),
    '/', subset.shape[0])

    predicates = []
    for k in range(mu.shape[0]):
        # if r[k] < 1.0 * (x[:,k].max()-x[:,k].min()):
        if True:
            r_k = (r[k] * x_std[k]).item()
            mu_k = (mu[k] * x_std[k] + x_mean[k]).item()
            ci = ((mu_k-r_k), (mu_k+r_k))
            predicates.append(dict(
                dim=k, interval=ci
            ))
    return dict(
        predicates=predicates
    )

embedding=None
@app.route('/get_embedding', methods=['GET'])
def get_embedding():
    return {
        'shape': embedding.shape,
        'value': b64encode(embedding.astype(np.float32).tobytes()).decode()
    }



df = pd.read_csv('./dataset/gait.csv')
x0 = df.to_numpy()
n_points = 10*101*6
df = df[:n_points]
x0 = x0[:n_points]
@app.route('/compute_predicates', methods=['POST'])
def compute_predicates():
    subset = np.array(request.json['subset'])
    predicates = predicate(x0, subset)
    print(predicates)
    return predicates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--embedding_fn',
        required=True,
        help='embedding file')

    opt = parser.parse_args()
    print(opt)

    embedding = np.load(opt.embedding_fn)

    app.run(host='127.0.0.1', port=9001, debug=True)
