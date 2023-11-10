# import pandas as pd
import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from typing import Union, Sequence
# from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pickle

# Нейросеть
class NNet(nn.Module):
    def __init__(self,
                 n_inputs: int,
                 n_layers: int,
                 n_neurons: Union[Sequence[int], int],
                 bias: bool=True):
        super().__init__()
        if not isinstance(n_inputs, int) or not isinstance(n_layers, int):
            raise TypeError("Введенные значения должны быть целыми числами")

        if len(n_neurons) != n_layers:
            raise ValueError("Количество нейронов должно быть равно числу слоев")
        n_neurons = [n_neurons] if isinstance(n_neurons, int) else n_neurons
        self.channels = [(n_inputs, n_neurons[0])] + \
                        [(n_neurons[i], n_neurons[i+1]) for i in range(len(n_neurons)-1)]
        self.net = nn.ModuleList([nn.Linear(*cnls, bias=bias) for cnls in self.channels])

    def forward(self, x):
        x = torch.from_numpy(x).float()
        for layer in self.net[:-1]:
            x = torch.sigmoid(layer(x))
        return self.net[-1](x)


# Модель
class TriticaleModel:
    def __init__(self,
                 n_inputs: int,
                 n_layers: int,
                 n_neurons: Union[Sequence[int], int],
                 bias: bool=True,
                 lr: float=0.1,
                 random_seed: int=None):
        self.random_seed = torch.manual_seed(random_seed) if random_seed else random_seed
        self.model = NNet(n_inputs, n_layers, n_neurons, bias)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def load_model(self, model_path: str=None) -> None:
        """
        Метод загрузка параметров модели
        """
        self.model.load_state_dict(torch.load(model_path))

    def train(self, X_train, y_train, X_val, y_val, n_epochs: int, save_path: str=None):
        curr_loss = float('inf')
        train_loss = []
        valid_loss = []
        y_train = torch.from_numpy(y_train.values).float().unsqueeze(1)
        y_val = torch.from_numpy(y_val.values).float().unsqueeze(1)

        for _ in range(n_epochs):
            # обучение
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(X_train)
            loss = self.loss_fn(output, y_train)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            # валидация
            self.model.eval()
            with torch.no_grad():
                output = self.model(X_val)
                loss = self.loss_fn(output, y_val)
                valid_loss.append(loss.item())
                # Сохраняем параметры лучшей модели
                if loss.item() < curr_loss and save_path:
                    torch.save(self.model.state_dict(), save_path)
                    curr_loss = loss.item()
        return train_loss, valid_loss

    def predict(self, X, y):
        """
        Метод предсказания и получения ошибки прогноза
        """
        self.model.eval()
        y = torch.tensor(y).unsqueeze(0)
        with torch.no_grad():
            output = self.model(X)
            loss = self.loss_fn(output, y).item()
        return output.item(), loss

    def get_all_predictions(self, X, y, percent=True, print_tab=True):
        """
        Метод получения предсказаний и ошибок по всем данным
        Опция вывода таблицы полученных значений
        """
        from tabulate import tabulate

        predictions = {}
        for i in range(X.shape[0]):
            y_pred, loss = self.predict(X[i, :], y[i])
            predictions.setdefault('Year', []).append(2010 + i)
            predictions.setdefault('Predictions', []).append(y_pred)
            predictions.setdefault('Absolute Error', []).append(loss ** 0.5)
            predictions.setdefault('Relative Error', []).append(round(loss ** 0.5 / y[i], 3))
        if percent:
            to_percent = lambda x: str(round(100 * x, 1)) + ' %'
            predictions['Relative Error'] = list(map(to_percent, predictions['Relative Error']))
        if print_tab:
            print(tabulate(list(zip(*predictions.values())), headers=predictions.keys()))
        return predictions

    @staticmethod
    def relative_error(y_pred, y_true):
        """
        Метод вычисления относительной ошибки прогноза
        """
        return abs(y_pred - y_true) / y_true


# Функции для отрисовки графиков
# Обучение
def learning_plot(train_loss, valid_loss):
    plt.figure(figsize=(12, 8))
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('Ошибка нейросети во время обучения и валидации', fontsize=20)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('MSE', fontsize=15)
    plt.legend(['train', 'valid'], fontsize='large')
    plt.show()


# Гистограмма результатов
def barplot(x, y_pred, target, target_name):
    plt.figure(figsize=(12, 8))
    index = np.arange(10)
    bw = 0.3
    plt.axis([-0.3, 10, 0, 80])
    plt.title(f'Эксперимент и предсказание для культуры {target_name}', fontsize=20)
    plt.bar(index, target, bw, color='b')
    plt.bar(index+bw, y_pred, bw, color='g')
    plt.ylabel('Урожайность, ц/га', fontsize=20)
    plt.yticks(fontsize=15)
    plt.xticks(index+bw/2, x, fontsize=15)
    plt.legend(['Эксперимент', 'Предсказание'], fontsize='large')
    plt.show()


def write_list(a_list, path, file_name='listfile'):
    """
    Записать список в бинарный файл.
    Cохранить список в двоичном файле, поэтому режим «wb».
    """
    with open(path + file_name, 'wb') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')


def read_list(path, file_name='listfile'):
    """
    Считать список в оперативную память.
    Для чтения также используется бинарный режим.
    """
    with open(path + file_name, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list