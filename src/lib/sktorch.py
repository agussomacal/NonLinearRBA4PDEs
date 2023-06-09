import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


def get_activation_function(activation_name):
    # https://towardsdatascience.com/pytorch-layer-dimensions-what-sizes-should-they-be-and-why-4265a41e01fd
    if activation_name.lower() in ["sigmoid", "logistic"]:
        func = torch.nn.Sigmoid()
        gain = torch.nn.init.calculate_gain('sigmoid')
    elif activation_name.lower() in ["relu"]:
        func = torch.nn.ReLU()
        gain = torch.nn.init.calculate_gain('relu')
    elif activation_name.lower() in ["tanh"]:
        func = torch.nn.Tanh()
        gain = torch.nn.init.calculate_gain('tanh')
    else:
        raise Exception('Activation function {} not implemented.'.format(activation_name))
    return func, gain


class FNNModel(torch.nn.Module, BaseEstimator, TransformerMixin):

    def __init__(self, hidden_layer_sizes, epochs=1000, activation='sigmoid', validation_size=0.2,
                 solver=torch.optim.Adam, lr=0.01,
                 lr_lower_limit=1e-12, lr_upper_limit=1, n_epochs_without_improvement=100, random_state=42, dropout_p=0,
                 batch_normalization=False, save_stats=False, loss_func=torch.nn.MSELoss(), no_improvement=50):
        super(FNNModel, self).__init__()
        self.loss_func = loss_func
        self.lr = lr
        self.validation_size = validation_size
        self.solver = solver
        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_p = dropout_p
        self.batch_normalization = batch_normalization
        self.epochs = epochs
        self.input_shape = None
        self.output_shape = None
        self.model = None

        self.no_improvement = no_improvement
        self.losses = {"train": [], "valid": []}

    def architecture(self):
        func, gain = get_activation_function(self.activation)

        prev_shape = list(self.input_shape) + list(self.hidden_layer_sizes)
        post_shape = list(self.hidden_layer_sizes) + list(self.output_shape)

        sequence = list()
        for i, (ishape, oshape) in enumerate(zip(prev_shape, post_shape)):

            linear = torch.nn.Linear(ishape, oshape, bias=True)
            torch.nn.init.xavier_uniform_(linear.weight, gain=gain)

            # --- Linear transformation (axons)
            sequence.append(linear)

            # --- Dropout
            if self.dropout_p > 0:
                sequence.append(torch.nn.Dropout(p=self.dropout_p))

            # --- applying batch norm
            if self.batch_normalization:
                sequence.append(torch.nn.BatchNorm1d(oshape))

            # --- only add activation function if it is not the outputlayer
            if i < len(prev_shape) - 1:
                sequence.append(func)

        return torch.nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x * 10000.0) / 10000.0

    def fit(self, X, y):
        self.input_shape = np.shape(X)[1:]
        self.output_shape = np.shape(y)[1:]
        self.model = self.architecture()

        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))
        validation_final_ix = int(len(X) * self.validation_size)

        self.optimizer = self.solver(self.parameters(), lr=self.lr)
        self.train()
        minloss = np.inf
        ix = 0
        for epoch in range(0, self.epochs):
            def closure():
                # Zero gradients, perform a backward pass,
                # and update the weights.
                self.optimizer.zero_grad()

                # Compute and print loss
                loss = self.loss_func(self.forward(X[validation_final_ix:]), y[validation_final_ix:])
                loss.backward()
                self.losses["train"].append(loss.item())
                return loss

            self.optimizer.step(closure)

            loss_valid = self.loss_func(self.forward(X[:validation_final_ix]), y[:validation_final_ix])
            self.losses["valid"].append(loss_valid.item())
            if minloss > loss_valid:
                minloss = loss_valid
                ix = epoch
            elif (epoch - ix) > self.no_improvement:  # early stopping by no improvement.
                print(f"\rEarly stopping in epoch {epoch} by no improvement.")
                break
            print('\repoch {}: loss valid {}'.format(epoch, loss_valid.item()))
        self.eval()

    def predict(self, X):
        return self.forward(torch.from_numpy(X.astype(np.float32))).detach().numpy()
