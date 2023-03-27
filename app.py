import torch
from torch import nn
from torch import optim

from flask import Flask
from flask import request
from flask_cors import CORS

import numpy as np
import pandas as pd

from base64 import b64encode
# from tqdm import tqdm

# create Flask app
app = Flask(__name__)
CORS(app)


def predict(x, a, mu):
    '''UMAP-inspired predict function
    x - torch tensor, shape [n_data_points, n_features]
    a - torch tensor, shape [n_features]
        1/a.abs() is the extent of bounding box at prediction=0.5
    mu - torch tensor, shape [n_features]
    b - scalar. hyper parameter for predict function. Power exponent
    '''

    b = 3
    return 1 / (1 + ((a.abs() * (x - mu).abs()).pow(b)).sum(1))

# test: UMAP-inspired predict function
# n = 100
# x = torch.linspace(-3,3,n).view(n,1)
# a = torch.tensor(0.5)
# plt.plot(x, predict(x, a))


def compute_predicate(x0, selected, n_iter=1000, mu_init=None, a_init=0.4):
    '''
        x0 - numpy array, shape=[n_points, n_feature]. Data points
        selected - boolean array. shape=[n_points] of selection
    '''

    # prepare training data
    # orginal data extent
    n_points, n_features = x0.shape
    vmin = x0.min(0)
    vmax = x0.max(0)
    x = torch.from_numpy(x0.astype(np.float32))
    label = torch.from_numpy(selected).float()
    # normalize
    mean = x.mean(0)
    scale = x.std(0) + 0.1
    x = (x - mean) / scale

    # Trainable parameters
    # since data is normalized,
    # mu can initialized around mean_pos examples
    # a can initialized around a constant across all axes
    center_selected = x[selected].mean(0)
    if mu_init is None:
        mu_init = center_selected
    a = (a_init + 0.1*(2*torch.rand(n_features)-1))
    mu = mu_init + 0.1 * (2*torch.rand(x.shape[1]) - 1)
    a.requires_grad_(True)
    mu.requires_grad_(True)

    # weight-balance selected vs. unselected based on their size
    n_selected = selected.sum()
    n_unselected = n_points - n_selected
    instance_weight = torch.ones(x.shape[0])
    instance_weight[selected] = n_points/n_selected
    instance_weight[~selected] = n_points/n_unselected
    bce = nn.BCELoss(weight=instance_weight)
    optimizer = optim.SGD([
        {'params': mu, 'weight_decay': 0},
        # smaller a encourages larger reach of the bounding box
        {'params': a, 'weight_decay': 0.01}
    ], lr=1e-2, momentum=0.9)

    # training loop
    for e in range(n_iter):
        pred = predict(x, a, mu)
        loss = bce(pred, label)
        loss += (mu - center_selected).pow(2).mean() * 20
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if e % (n_iter//5) == 0:
            # print(pred.min().item(), pred.max().item())
            print(f'[{e:>4}] loss {loss.item()}')
    a.detach_()
    mu.detach_()
    # plt.stem(a.abs().numpy()); plt.show()

    pred = (pred > 0.5).float()
    correct = (pred == label).float().sum().item()
    total = selected.shape[0]
    accuracy = correct/total
    # 1 meaning points are selected
    tp = ((pred == 1).float() * (label == 1).float()).sum().item()
    fp = ((pred == 1).float() * (label == 0).float()).sum().item()
    fn = ((pred == 0).float() * (label == 1).float()).sum().item()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 1/(1/precision + 1/recall)
    print(f'''
accuracy = {correct/total}
precision = {precision}
recall = {recall}
f1 = {f1}
    ''')

    # predicate clause selection
    # r is the range of the bounding box on each dimension
    # bounding box is defined by the level set of prediction=0.5
    r = 1 / a.abs()
    predicates = []
    for k in range(mu.shape[0]):
        # denormalize
        r_k = (r[k] * scale[k]).item()
        mu_k = (mu[k] * scale[k] + mean[k]).item()
        ci = [mu_k - r_k, mu_k + r_k]
        assert ci[0] < ci[1], 'ci[0] is not less than ci[1]'
        if ci[0] < vmin[k]:
            ci[0] = vmin[k]
        if ci[1] > vmax[k]:
            ci[1] = vmax[k]
        # feature selection based on extent range
#         should_include = r[k] < 1.0 * (x[:,k].max()-x[:,k].min())
        should_include = not (ci[0] <= vmin[k] and ci[1] >= vmax[k])
        if should_include:
            predicates.append(dict(
                dim=k, interval=ci
            ))
    for p in predicates:
        print(p)
    return predicates, mu, a, accuracy, precision, recall, f1


current_dataset = None
x0 = None


@app.route('/get_predicates', methods=['POST'])
def get_predicate():
    global current_dataset, x0
    dataset = request.json['dataset']
    # load dataset csv
    if current_dataset != dataset:
        df = pd.read_csv(f'./dataset/{dataset}.csv')
        for attr in ['x', 'y', 'image_filename']:
            if attr in df.columns:
                df = df.drop(attr, axis='columns')
        x0 = df.to_numpy()
        current_dataset = dataset

    # a sequence of bool arrays indexed by [brush time, data point index]
    subsets = np.array(request.json['subsets'])

    predicates = []
    quality_measures = []
    mu = None
    a = 0.4
    for subset in subsets:
        predicate, mu, a, accuracy, precision, recall, f1 = compute_predicate(
            x0, subset, mu_init=mu, a_init=a)
        predicates.append(predicate)
        quality_measures.append(dict(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
        ))
    # TODO compute and report precision recall accuracy and  F1
    return dict(
        predicates=predicates,
        quality_measures=quality_measures,
    )


# embedding = None
# @app.route('/get_embedding', methods=['GET'])
# def get_embedding():
#     return {
#         'shape': embedding.shape,
#         'value': b64encode(embedding.astype(np.float32).tobytes()).decode()
#     }


# df = pd.read_csv('./dataset/gait_joined.csv')
# x0 = df.to_numpy()


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--embedding_fn',
    #     required=True,
    #     help='embedding file')
    # opt = parser.parse_args()
    # print(opt)
    # embedding = np.load(opt.embedding_fn)

    app.run(host='127.0.0.1', port=9001, debug=True)
