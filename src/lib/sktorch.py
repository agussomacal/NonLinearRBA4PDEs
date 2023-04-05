import numpy as np
import seaborn as sns
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

# -

df = sns.load_dataset("tips")
df.head()


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
                 batch_normalization=False, save_stats=False, loss_func=torch.nn.MSELoss()):
        super(FNNModel, self).__init__()
        self.loss_func = loss_func
        self.lr = lr
        self.solver = solver
        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_p = dropout_p
        self.batch_normalization = batch_normalization
        self.epochs = epochs
        self.input_shape = None
        self.output_shape = None
        self.model = None

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
        return self.model(x)

    def fit(self, X, y):
        self.input_shape = np.shape(X)[1:]
        self.output_shape = np.shape(y)[1:]
        self.model = self.architecture()

        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        self.optimizer = self.solver(self.parameters(), lr=self.lr)
        self.train()
        for epoch in tqdm(range(0, self.epochs)):
            pred_y = self.forward(X)

            # Compute and print loss
            loss = self.loss_func(pred_y, y)

            # Zero gradients, perform a backward pass,
            # and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('epoch {}, loss {}'.format(epoch, loss.item()))
        self.eval()

    def predict(self, X):
        return self.forward(torch.from_numpy(X.astype(np.float32))).detach().numpy()


# class FNNOverKModel(FNNModel):
#
#     def __init__(self, hidden_layer_sizes, freq=-1, epochs=1000, activation='sigmoid', validation_size=0.2,
#                  solver=torch.optim.Adam, lr=0.01,
#                  lr_lower_limit=1e-12, lr_upper_limit=1, n_epochs_without_improvement=100, random_state=42, dropout_p=0,
#                  batch_normalization=False, save_stats=False, loss_func=torch.nn.MSELoss()):
#         super(FNNOverKModel, self).__init__(
#             hidden_layer_sizes=hidden_layer_sizes, epochs=epochs, activation=activation,
#             validation_size=validation_size, solver=solver, lr=lr, lr_lower_limit=lr_lower_limit,
#             lr_upper_limit=lr_upper_limit, n_epochs_without_improvement=n_epochs_without_improvement,
#             random_state=random_state, dropout_p=dropout_p, batch_normalization=batch_normalization,
#             save_stats=save_stats, loss_func=loss_func)
#         self.power = power
#
#     def architecture(self):
#         model = super(FNNOverKModel, self).architecture()
#         self.output_shape
