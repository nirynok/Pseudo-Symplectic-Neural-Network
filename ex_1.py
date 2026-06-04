import torch
from torch import optim
from torch import autograd
from torch import nn
import torch.nn.functional
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import argparse
import h5py
import pickle
import os
import utils


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--epochs', default=1500, type=int, help='number of epochs')
    parser.add_argument('--training_samples', default=15, type=int, help='number of training samples')
    parser.add_argument('--testing_samples', default=100, type=int, help='number of testing samples')
    parser.add_argument('--nx', default=4, type=int, help='nx')
    parser.add_argument('--ny', default=32, type=int, help='ny')
    parser.add_argument('--L1', default=3, type=int, help='L1')
    parser.add_argument('--L2', default=1, type=int, help='L2')
    parser.add_argument('--h', default=0.01, type=float, help='h')
    parser.add_argument('--T', default=0.01, type=float, help='T')
    parser.add_argument('--gen', default=1, type=int, help='gen')
    parser.add_argument('--batch', default=1, type=int, help='batch')
    parser.add_argument("--activation", type=str, default="pade_type")
    parser.add_argument('--data', default='_psnn', type=str)
    parser.add_argument("--save", type=str, default="experiments")
    parser.set_defaults(feature=True)
    return parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
J = torch.tensor([[0., 1.], [-1., 0.]])
h = 0.01
d = 2 # dimension


def hamiltonian(x):
    # 1
    def U(q):
        return 0.1 * (q * (q - 1))
        
    def g_U(q):
        return 0.1 * (2 * q - 1)

    H = x[0] ** 2 / (2 * (1 + g_U(x[1]) ** 2)) + U(x[1])

    return H


# gradient of hamiltonian
def grad_h(x):
    x.requires_grad_()
    H = hamiltonian(x)
    g_h = autograd.grad(outputs=H, inputs=x, grad_outputs=torch.ones_like(H), create_graph=True, retain_graph=True)[0]
    return (g_h@J).detach()


def solve_ode(initial_condition, t0, t1, time_steps):

    def ode_func(y):
        return grad_h(y)

    solution = torch.zeros((time_steps + 1, d))
    solution[0] = initial_condition

    dt = (t1 - t0) / time_steps

    for i in range(1, time_steps + 1):

        y_n = solution[i - 1]

        y_k = y_n.clone()

        for _ in range(50):

            y_new = y_n + dt * ode_func((y_n + y_k) / 2)

            if torch.norm(y_new - y_k) < 1e-14:
                y_k = y_new
                break

            y_k = y_new

        solution[i] = y_k

    return solution


def gen_data(args, n_samples, time_steps, data_type='train'):
    y0s = []
    y1Ts = []

    sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True)
    samples = sobol.draw(n_samples)
    samples = 4 * samples - 2

    for i in range(n_samples):
        t0 = torch.tensor(0.)
        t1 = torch.tensor(args.T)
        y0 = samples[i]
        y1T = solve_ode(y0, t0, t1, time_steps)
        y0s.append(y0.view(1, d))
        y1Ts.append(y1T[-1].view(1, d))
    y0s = torch.cat(y0s).detach().cpu().numpy()
    y1Ts = torch.cat(y1Ts).detach().cpu().numpy()

    data_root = os.path.join(os.path.dirname(os.path.realpath(__file__)))

    hf = h5py.File(os.path.join(data_root, data_type + args.data + ".h5"), "w")
    hf.create_dataset('y0', data=y0s)
    hf.create_dataset('y1T', data=y1Ts)
    hf.close()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_type, data):
        datafile = os.path.join(os.path.dirname(os.path.realpath(__file__)), data_type+data+'.h5')
        f = h5py.File(datafile)
        self.y0 = f['y0'][:]
        self.y1T = f['y1T'][:]
        f.close()

        self.y0, self.y1T = self.y0.astype(np.float32), self.y1T.astype(np.float32)

    def __getitem__(self, index):
        return self.y0[index], self.y1T[index]

    def __len__(self):
        return self.y0.shape[0]


# PSNN
class Net(nn.Module):
    def __init__(self, nx, ny, L1, L2):
        super(Net, self).__init__()
        self.L1 = L1
        self.L2 = L2
        self.nx = nx
        self.ny = ny
        l = 1/(4*(2-2**(1/3)))

        A = torch.tensor([
            [0., 0., 0., 0., 0., 0., 0.],
            [2*l, 0., 0., 0., 0., 0., 0.],
            [0., 4*l, 0., 0., 0., 0., 0.],
            [2*l, 0., 1/2-2*l, 0., 0., 0., 0.],
            [0., 4*l, 0., 1-8*l, 0., 0., 0.],
            [2*l, 0., 1/2-2*l, 0., 1/2-2*l, 0., 0.],
            [0., 4*l, 0., 1-8*l, 0., 4*l, 0.]
        ])

        B = torch.tensor([
            l,
            2*l,
            1/4-l,
            1/2-4*l,
            1/4-l,
            2*l,
            l
        ])

        self.register_buffer("A", A)
        self.register_buffer("B", B)

        self.register_buffer("J", torch.tensor([[0.,1.],[-1.,0.]]))

        self.p = nn.Parameter(0.05*torch.randn(self.nx,self.L1+1))

        self.q = nn.Parameter(0.05*torch.rand(self.nx,self.L2+1))

        self.K1 = nn.Parameter(torch.empty(self.nx,d,self.ny))

        self.K2 = nn.Parameter(torch.empty(self.nx,d,self.ny))

        nn.init.xavier_uniform_(self.K1)
        nn.init.xavier_uniform_(self.K2)

        self.b = nn.Parameter(0.01*torch.randn(1,2))

        
    
    def pade_batch(self, z):

        # numerator

        num = self.p[:, self.L1].view(1, self.nx, 1)

        for j in range(self.L1 - 1, -1, -1):

            num = (num * z + self.p[:, j].view(1, self.nx, 1))

        # denominator

        den = self.q[:, self.L2].view(1, self.nx, 1)

        for j in range(self.L2 - 1, -1, -1):

            den = (den * z + self.q[:, j].view(1, self.nx, 1))

        den = den * den + 1.0

        return num / den
        


    def forward(self,x):

        z2 = torch.einsum('bd,ndm->bnm', x, self.K2)

        z1 = torch.einsum('bd,ndm->bnm', x, self.K1)

        phi2 = self.pade_batch(z2)
        phi1 = self.pade_batch(z1)

        g2 = torch.einsum('bnm,ndm->bnd', phi2, self.K2).sum(dim=1)

        g1 = torch.einsum('bnm,ndm->bnd', phi1, self.K1).sum(dim=1)

        g = g2 - g1

        g = g + self.b

        return g @ self.J


    def psnn(self,x,h):

        a = self.A
        b = self.B

        k = [None]*7

        k[0] = self(x)

        for i in range(6):

            grad = 0

            for j in range(i+1):
                grad = grad + a[i+1,j]*k[j]

            k[i+1] = self(x + h*grad)

        update = 0

        for i in range(7):
            update = update + b[i]*k[i]

        return x + h*update

    def psnn_T(self, x, h, T):
        N = int(T/h)
        for _ in range(N):
            x = self.psnn(x, h)
        return x


if __name__ == "__main__":
    args = get_args()
    utils.makedirs(args.save)
    logger = utils.get_logger(
        logpath=os.path.join(args.save, "%s_logs_%d_%d_%d_%d_%f_%f_%s.txt") % (
            args.activation, args.nx, args.ny, args.L1, args.L2, args.h, args.learning_rate, args.data),
        filepath=os.path.abspath(__file__)
    )
    # generate training data and testing data
    if args.gen == 1:
        gen_data(args, n_samples=args.training_samples, data_type='train', time_steps=int(args.T/0.001))
        gen_data(args, n_samples=args.testing_samples, data_type='test', time_steps=int(args.T / 0.001))

    data_loader_train = torch.utils.data.DataLoader(Dataset(data_type='train', data=args.data),
                                                    batch_size=args.training_samples, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(Dataset(data_type='test', data=args.data),
                                                   batch_size=args.testing_samples,
                                                   shuffle=True)

    model = Net(args.nx, args.ny, args.L1, args.L2).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_loss_s = []
    test_loss_s = []

    for epoch in range(args.epochs):
        train_loss = 0
        train_sample = 0
        for batch_index, data_batch in enumerate(data_loader_train):
            x, y = data_batch
            x = x.to(device)
            y = y.to(device)

            outputs = model.psnn_T(x, args.h, args.T)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.detach().cpu().item()
            train_sample += 1
            optimizer.step()

        train_loss_s.append(train_loss)
        
        if (epoch + 1) % 50 == 0:
            test_loss = 0
            test_sample = 0
            with torch.no_grad():
                for batch_index, data_batch in enumerate(data_loader_test):
                    x, y = data_batch
                    x = x.to(device)
                    y = y.to(device)
                    outputs = model.psnn_T(x, args.h, args.T)
                    loss = criterion(outputs, y)
                    test_loss += loss.detach().cpu().item()
                    test_sample += 1
            test_loss_s.append(test_loss)
            logger.info(
                f"Epoch [{epoch + 1}/{args.epochs}], training loss: {train_loss / train_sample}, testing loss: {test_loss / test_sample}")
    file = open(
        './experiments/{}_{}_{}_{}_{}_{}_{}_{}_training_loss.pkl'.format(args.activation, args.nx, args.ny, args.L1, args.L2, args.h,
                                                                          args.learning_rate, args.data), 'wb')
    pickle.dump(train_loss_s, file)
    file.close()
    file = open(
        './experiments/{}_{}_{}_{}_{}_{}_{}_{}_training_loss2.pkl'.format(args.activation, args.nx, args.ny, args.L1, args.L2,
                                                                               args.h,
                                                                               args.learning_rate, args.data), 'wb')
    pickle.dump(test_loss_s, file)
    file.close()
    file = open(
        './experiments/{}_{}_{}_{}_{}_{}_{}_{}_testing_loss2.pkl'.format(args.activation, args.nx, args.ny, args.L1, args.L2,
                                                                               args.h,
                                                                               args.learning_rate, args.data), 'wb')

    # prediction
    n_p = 5000
    t = torch.linspace(0, n_p * h, n_p + 1)
    # true
    t0 = torch.tensor(0.)
    t1 = torch.tensor(50.)
    x = torch.tensor([[0., 1.]]) #initial point
    X = solve_ode(x, t0, t1, n_p*10) #true solution

    # PSNN prediction
    x_psnn = torch.tensor([[0., 1.]], device=device) #initial point
    X_psnn = torch.zeros(n_p + 1, d, device=device)
    X_psnn[0] = x_psnn[0]

    with torch.no_grad():
        for i in range(1, n_p + 1):
            x_psnn = model.psnn(x_psnn, h)
            X_psnn[i, :] = x_psnn[0].detach()

    X_psnn = X_psnn.cpu()

    # plot
    plt.figure(figsize=(8, 6))
    
    plt.scatter(X[:, 0], X[:, 1], s=2, c='r', label='True')

    plt.scatter(X_psnn[:, 0], X_psnn[:, 1], s=2, c='green', label='Predicted by PSNN')
    
    plt.xlabel('p')
    plt.ylabel('q')
    plt.legend()
    plt.savefig("./experiments/{}_{}_{}_{}_{}_{}_{}_{} Prediction.png".format(args.activation, args.nx, args.ny, args.L1, args.L2, args.h,
                                                                          args.learning_rate, args.data))
    plt.show()






