import tools
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from scipy.interpolate import interp1d
import os
import re


def load_data():
    x_dict = tools.get_all_surfacelines('rijnland')
    y_dict = tools.get_all_cpoints('rijnland')

    xy = [np.array(list(map(lambda x: [x[1], x[2]], p))) for p in [x_dict[k] for k in x_dict.keys()]]

    # clicked points
    clicked = [np.array(list(map(lambda x: [x[0], x[1], x[2]], p))) for p in [y_dict[k] for k in x_dict.keys()]]
    return xy, clicked


def scale_features(x):
    return (x - x.min(0)) / (x.max(0) - x.min(0))


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)


def train_model(m, x_train, y_train):
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(m.parameters(), lr=0.001)

    epochs = 2_500
    batch_size = 50
    m.train(True)

    idx = np.arange(len(x_train))
    x_train = torch.tensor(x_train).cuda().float()
    y_train = torch.tensor(y_train).cuda().float()
    first_iteration = True

    for epoch in tqdm(range(epochs)):
        np.random.shuffle(idx)
        x_train = x_train[idx]
        y_train = y_train[idx]

        current_batch = 0
        for iteration in range(y_train.shape[0] // batch_size):
            batch_x = x_train[current_batch: current_batch + batch_size]
            batch_y = y_train[current_batch: current_batch + batch_size]
            current_batch += batch_size

            if len(batch_x) > 0:
                batch_pred = m(batch_x)
                optim.zero_grad()

                loss = criterion(batch_pred, batch_y)
                loss.backward()
                optim.step()
                if first_iteration:
                    first_loss = loss.item()
                    first_iteration = False

    print(f'Starting loss: {first_loss}, \t Final loss: {loss.item()}')
    return m


class Inference:
    def __init__(self, n_interpolations=50, training_points=(5, 8, 9, 10, 11)):
        self.x = np.linspace(0, 1, n_interpolations)
        self.xy, self.clicked = load_data()
        # self.xy = list(map(scale_features, xy))
        self.training_points = training_points
        self.feat = None
        self.feature_length = n_interpolations + n_interpolations - 1
        self.labels = None
        self.models = None
        self.data = None  # scaled data [mask label, x, y]

    def prepare_label(self, label):
        labels = []
        for i in range(len(self.xy)):
            mask = np.isin(self.clicked[i][:, 0], np.array([label]))
            idx = [np.argmin(np.sum(np.abs(self.xy[i] - p), 1), axis=0) for p in self.clicked[i][mask][:, 1:]]
            l = np.zeros(self.xy[i].shape[0], dtype=bool)
            l[idx] = True
            labels.append(l)
        return labels

    def prepare_inference_data(self, point_label=None):
        labels = self.prepare_label(point_label)
        xy = map(scale_features, self.xy)
        data = list(map(lambda t: np.hstack((t[1][:, None], t[0])), zip(xy, labels)))

        if self.feat is None:
            # features are the y values of the points. As the x values are always the same due to interpolation.
            feat = np.stack(list(map(lambda a: interp1d(a[:, 1], a[:, 2])(self.x), data)))

            # add difference
            self.feat = np.hstack((feat, np.diff(feat)))

        if point_label:
            # The true points that are closest
            labels = np.stack(list(map(lambda x: x[:, 1:][np.isclose(x[:, 0], 1, atol=1e-9)], data)))[:, -1]
            self.labels = labels

        self.data = data
        return self.feat, self.labels

    def train_models(self, hidden=120):
        c = 0
        n = len(self.training_points)
        for point_label in self.training_points:
            c += 1
            print(f'training model, {c}/{n}')
            self.prepare_inference_data(point_label)
            m = MLP(self.feat.shape[1], hidden, 2)
            m.cuda()
            a = 1200
            m = train_model(m, self.feat[:a], self.labels[:a]).cpu()
            torch.save(m.state_dict(), f'save/mlp_{point_label}.pth')

    def load_models(self, hidden=120):

        p = os.listdir('save')

        self.models = [0 for _ in range(len(self.training_points))]
        for path in p:
            g = re.search(r'mlp_(\d+).pth', path)
            nr = g.group(1)
            idx = list(self.training_points).index(int(nr))
            m = MLP(self.feature_length, hidden, 2)
            m.load_state_dict(torch.load(f'save/{path}'))
            m.train(False)
            self.models[idx] = m

    def inference(self, profile_idx):
        def distance(x1, y1, x2, y2):
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        self.prepare_inference_data()
        fig, axes = plt.subplots(2, 1, figsize=(6, 8))
        x = self.data[profile_idx][:, 1]
        y = self.data[profile_idx][:, 2]

        axes[0].set_title(profile_idx)
        axes[0].plot(x, y)
        axes[1].plot(x, y)
        for i in range(len(self.models)):
            point_label = self.training_points[i]

            self.prepare_inference_data(point_label)
            pred = self.models[i](torch.tensor(self.feat).float()[profile_idx][None, :])
            x_, y_ = tuple(pred.data.numpy().tolist()[0])
            idx = distance(x, y, x_, y_).argmin()

            axes[0].scatter(x_, y_, color='purple', alpha=0.2)

            axes[0].scatter(x[idx], y[idx], color='r', alpha=0.8)  # closest point
            axes[0].annotate(point_label, (x[idx], y[idx]))

            axes[1].scatter(self.labels[profile_idx][0], self.labels[profile_idx][1], color='green', alpha=0.8)
            axes[1].annotate(point_label, (self.labels[profile_idx][0], self.labels[profile_idx][1]))

        plt.show()


TRAIN = False

a = Inference()

if TRAIN:
    a.train_models(120)
else:
    a.load_models()

    # for i in (1292, 1300, 1305, 1309, 12, 23, 50, 70):
    a.inference(1201)

