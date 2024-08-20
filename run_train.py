# Standard importations
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import os
import sys
import numpy as np
import argparse
import logging
import string
import random
from torch_ema import ExponentialMovingAverage
# For Time measurement
from datetime import datetime
from time import time
# Neural Network importations
from cleaning.Submit.Neural_Net import PhysNet
from layers.utils import segment_sum
from layers.activation_fn import *
from cleaning.Submit.Neural_Net import gather_nd
from DataContainer import DataContainer
#Other importations
# Configure logging environment
logging.basicConfig(filename='train.log', level=logging.DEBUG)

# ------------------------------------------------------------------------------
# Command line arguments
# ------------------------------------------------------------------------------

# Initiate parser
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

# Add arguments
parser.add_argument("--restart", type=str, default='No',
                    help="Restart training from a specific folder")
parser.add_argument("--checkpoint-file", type=str, default=None,
                    help="File to be loaded if model is restarted")
parser.add_argument("--num_features", default=128, type=int,
                    help="Dimensionality of feature vectors")
parser.add_argument("--num_basis", default=64, type=int,
                    help="Number of radial basis functions")
parser.add_argument("--num_blocks", default=5, type=int,
                    help="Number of interaction blocks")
parser.add_argument("--num_residual_atomic", default=2, type=int,
                    help="Number of residual layers for atomic refinements")
parser.add_argument("--num_residual_interaction", default=3, type=int,
                    help="Number of residual layers for the message phase")
parser.add_argument("--num_residual_output", default=1, type=int,
                    help="Number of residual layers for the output blocks")
parser.add_argument("--cutoff", default=10.0, type=float,
                    help="Cutoff distance for short range interactions")
parser.add_argument("--use_electrostatic", default=1, type=int,
                    help="Use electrostatics in energy prediction (0/1)")
parser.add_argument("--use_dispersion", default=1, type=int,
                    help="Use dispersion in energy prediction (0/1)")
parser.add_argument("--grimme_s6", default=None, type=float,
                    help="Grimme s6 dispersion coefficient")
parser.add_argument("--grimme_s8", default=None, type=float,
                    help="Grimme s8 dispersion coefficient")
parser.add_argument("--grimme_a1", default=None, type=float,
                    help="Grimme a1 dispersion coefficient")
parser.add_argument("--grimme_a2", default=None, type=float,
                    help="Grimme a2 dispersion coefficient")
parser.add_argument("--dataset", type=str,
                    help="File path to dataset")
# This number is configured for the size of the QM9 dataset
parser.add_argument("--num_train", default=103130, type=int,
                    help="Number of training samples")
# This number is configured for the size of the QM9 dataset
parser.add_argument("--num_valid", default=12891, type=int,
                    help="Number of validation samples")
parser.add_argument("--batch_size", default=100, type=int,
                    help="Batch size used per training step")
parser.add_argument("--valid_batch_size", default=20, type=int,
                    help="Batch size used for going through validation_set")
parser.add_argument("--seed", default=np.random.randint(1000000), type=int,
                    help="Seed for splitting dataset into " + \
                         "training/validation/test")
parser.add_argument("--max_steps", default=10000, type=int,
                    help="Maximum number of training steps")
parser.add_argument("--learning_rate", default=0.001, type=float,
                    help="Learning rate used by the optimizer")
parser.add_argument("--decay_steps", default=1000, type=int,
                    help="Decay the learning rate every N steps by decay_rate")
parser.add_argument("--decay_rate", default=0.1, type=float,
                    help="Factor with which the learning rate gets " + \
                         "multiplied by every decay_steps steps")
parser.add_argument("--max_norm", default=1000.0, type=float,
                    help="Max norm for gradient clipping")
parser.add_argument("--ema_decay", default=0.999, type=float,
                    help="Exponential moving average decay used by the " + \
                         "trainer")
parser.add_argument("--rate", default=0.0, type=float,
                    help="Rate probability for dropout regularization of " + \
                         "rbf layer")
parser.add_argument("--l2lambda", default=0.0, type=float,
                    help="Lambda multiplier for l2 loss (regularization)")
#Note: This parameter is setup to 0.2 as it was the best value on the paper of Amini...
parser.add_argument("--lambda_conf", default=0.2, type=float,
                    help="Lambda value of the confidence of the prediction")
parser.add_argument('--summary_interval', default=5, type=int,
                    help="Write a summary every N steps")
parser.add_argument('--validation_interval', default=5, type=int,
                    help="Check performance on validation set every N steps")
parser.add_argument('--show_progress', default=True, type=bool,
                    help="Show progress of the epoch")
parser.add_argument('--save_interval', default=5, type=int,
                    help="Save progress every N steps")
parser.add_argument('--record_run_metadata', default=0, type=int,
                    help="Records metadata like memory consumption etc.")
parser.add_argument('--device',default='cuda',type=str,
                    help='Selects the device that will be used for training')
parser.add_argument('--DER_type',default=None,type=str,
                    help='Type of DER')

# ------------------------------------------------------------------------------
# Read Parameters and define output files
# ------------------------------------------------------------------------------


# Generate an (almost) unique id for the training session
def id_generator(size=8,
                 chars=(string.ascii_uppercase
                        + string.ascii_lowercase
                        + string.digits)):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))


# Read config file if no arguments are given
config_file = 'config.txt'
if len(sys.argv) == 1:
    if os.path.isfile(config_file):
        args = parser.parse_args(["@" + config_file])
    else:
        args = parser.parse_args(["--help"])
else:
    args = parser.parse_args()

# Create output directory for training session and
# load config file arguments if restart
if args.restart == 'No':
    directory = (
            datetime.utcnow().strftime("%Y%m%d%H%M%S")
            + "_" + id_generator() + "_F" + str(args.num_features)
            + "K" + str(args.num_basis) + "b" + str(args.num_blocks)
            + "a" + str(args.num_residual_atomic)
            + "i" + str(args.num_residual_interaction)
            + "o" + str(args.num_residual_output) + "cut" + str(args.cutoff)
            + "e" + str(args.use_electrostatic) + "d" + str(args.use_dispersion)
            + "rate" + str(args.rate))
    checkpoint_file = args.checkpoint_file
else:
    directory = args.restart
    args = parser.parse_args(["@" + os.path.join(args.restart, config_file)])
    checkpoint_file = os.path.join(args.restart, args.checkpoint_file)

# Create sub directories
logging.info("Creating directories...")

if not os.path.exists(directory):
    os.makedirs(directory)

best_dir = os.path.join(directory, 'best')
if not os.path.exists(best_dir):
    os.makedirs(best_dir)

log_dir = os.path.join(directory, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

ckpt_dir = os.path.join(directory, 'ckpt')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# Define output files
best_loss_file = os.path.join(best_dir, 'best_loss.npz')


# Write config file of current training session
logging.info("Writing args to file...")

with open(os.path.join(directory, config_file), 'w') as f:
    for arg in vars(args):
        f.write('--' + arg + '=' + str(getattr(args, arg)) + "\n")

logging.info("device: {}".format(args.device))

# ------------------------------------------------------------------------------
# Define utility functions
# ------------------------------------------------------------------------------
def save_checkpoint(model, epoch, name_of_ckpt=None, best=False):
    state = {'model_state_dict': model.state_dict(),
             'epoch': epoch}
    if best:
        path = os.path.join(best_dir, 'best_model.pt')
    else:
        name = 'model' + str(name_of_ckpt) + '.pt'
        path = os.path.join(ckpt_dir, name)
    torch.save(state, path)


def load_checkpoint(path):
    if path is not None:
        checkpoint = torch.load(path)
        return checkpoint
    else:
        return None


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='#', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print("\r{0} |{1}| {2}% {3}".format(
        prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def reset_averages(type,device='cpu'):
    ''' Reset counter and average values '''
    null_float = torch.tensor(0.0, dtype=torch.float32,device=device)
    if type == "train":
        return null_float, null_float, null_float, null_float, null_float, \
           null_float, null_float, null_float, null_float, null_float,null_float,null_float
    elif type == "valid":
        return null_float, null_float, null_float, null_float, null_float, \
           null_float, null_float, null_float, null_float, null_float

def l2_regularizer(model,l2_lambda=args.l2lambda):
    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
    return l2_lambda*l2_norm

#====================================
# Some functions
#====================================
def compute_pnorm(model):
    """Computes the norm of the parameters of a model."""
    return np.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


def compute_gnorm(model):
    """Computes the norm of the gradients of a model."""
    return np.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))
# ------------------------------------------------------------------------------
# Load data and initiate PhysNet model
# ------------------------------------------------------------------------------

# Load dataset
logging.info("Loading dataset...")

data = DataContainer(
    args.dataset, args.num_train, args.num_valid,
    args.batch_size, args.valid_batch_size, seed=args.seed)

# Initiate PhysNet model
logging.info("Creating PhysNet model...")

# Initiate summary writer
summary_writer = SummaryWriter(log_dir)
model = PhysNet(
    F=args.num_features,
    K=args.num_basis,
    sr_cut=args.cutoff,
    num_blocks=args.num_blocks,
    num_residual_atomic=args.num_residual_atomic,
    num_residual_interaction=args.num_residual_interaction,
    num_residual_output=args.num_residual_output,
    use_electrostatic=(args.use_electrostatic == 1),
    use_dispersion=(args.use_dispersion == 1),
    s6=args.grimme_s6,
    s8=args.grimme_s8,
    a1=args.grimme_a1,
    a2=args.grimme_a2,
    Eshift=data.EperA_m_n,
    Escale=data.EperA_s_n,
    activation_fn=shifted_softplus,
    device=args.device,
    writer=summary_writer)


if os.path.isfile(best_loss_file):
    loss_file = np.load(best_loss_file)
    best_loss = loss_file["loss"].item()
    best_emae = loss_file["emae"].item()
    best_ermse = loss_file["ermse"].item()
else:
    best_loss = np.Inf
    best_emae = np.Inf
    best_ermse = np.Inf
    best_epoch = 0.
    np.savez(
        best_loss_file, loss=best_loss, emae=best_emae, ermse=best_ermse,
        epoch=best_epoch)

# Print model
DER_type = args.DER_type

if DER_type is None:
    print('DER Type no especified, calculation will not be done')
    exit()
else:
    print('DER Type:{}'.format(DER_type))


#------------------------------------
# Loss function
#------------------------------------

def evid_loss_all(E_pred, v, alpha, beta, Q_pred, F_pred, D_pred,E_ref,Q_ref,F_ref,D_ref,
                  wf=52.9177, wq=14.3996, wd=27.2113,
                  lam=args.lambda_conf, epsilon=1e-4):
    """
    Use Deep Evidential Regression negative log likelihood loss + evidential
        regularizer
    We will use the new version on the paper..
    :mu: pred mean parameter for NIG
    :v: pred lam parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :E: Energies predict
    :F Forces predict
    :return: Loss
    """
    # Calculate NLL loss
    twoBlambda = 2*beta*(1+v)
    nll = 0.5*torch.log(np.pi/v) \
        - alpha*torch.log(twoBlambda) \
        + (alpha+0.5) * torch.log(v*(E_ref-E_pred)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha+0.5)

    L_NLL = nll #torch.mean(nll, dim=-1)

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((E_ref - E_pred))
    reg = error * (2 * v + alpha)
    L_REG = reg #torch.mean(reg, dim=-1)

    # Calculate error for the forces:
    LF = nn.L1Loss(reduction="mean")
    L_F = LF(F_ref, F_pred)

    # Calculate error for the charges as L1
    L_Q = LF(Q_ref, Q_pred)
    # Calculate error for the dipole moments as L1
    L_D = LF(D_ref, D_pred)

    # Complete loss
    loss = (L_NLL + lam * (L_REG - epsilon) + wf * L_F + wq * L_Q + wd * L_D
            + l2_regularizer(model))

    return loss

def lipschitz_loss(mu, v, alpha, beta, targets):
    lam_sqrt = torch.square(targets-mu)
    u_nu = (beta*(v+1))/(alpha*v)
    u_alpha = ((2*beta*(v+1))/v)*(torch.exp(torch.digamma(alpha+0.5)-torch.digamma(alpha))-1)
    u_min = torch.min(u_nu,u_alpha).min()
    delta = torch.abs(targets-mu)
    c = 2*torch.sqrt(u_min)*delta - u_min
    coeff = torch.clip(c,min=False,max=1.)
    return coeff*lam_sqrt
def lipz_loss_all(E_pred,nu,alpha,beta,Q_pred,F_pred,D_pred,E_ref,Q_ref,F_ref,D_ref,
                  wf=52.9177, wq=14.3996, wd=27.2113,
                  lam=args.lambda_conf, epsilon=1e-4):

    # This loss function calculates the loss for all properties of physnet and uses the
    # DER loss function + a Lipschitz loss for the E prediction

    # Calculate NLL loss
    twoBlambda = 2 * beta * (1 + nu)
    nll = 0.5 * torch.log(np.pi / nu) \
          - alpha * torch.log(twoBlambda) \
          + (alpha + 0.5) * torch.log(nu * (E_ref - E_pred) ** 2 + twoBlambda) \
          + torch.lgamma(alpha) \
          - torch.lgamma(alpha + 0.5)

    L_NLL = nll  # torch.mean(nll, dim=-1)

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((E_ref - E_pred))
    reg = error * (2 * nu + alpha)
    L_REG = reg  # torch.mean(reg, dim=-1)
    L_lipschitz_ener = lipschitz_loss(E_pred, nu, alpha, beta, E_ref)

    L_DER = L_NLL + lam * (L_REG - epsilon) + L_lipschitz_ener

    # Calculate error for the forces:
    LF = nn.L1Loss(reduction="mean")
    L_F = LF(F_ref, F_pred)
    # Calculate error for the charges as L1
    L_Q = LF(Q_ref,Q_pred)
    # Calculate error for the dipole moments as L1
    L_D = LF(D_ref,D_pred)

    # Complete loss

    loss = L_DER+ wf * L_F + wq * L_Q + wd * L_D + l2_regularizer(model)

    return loss

def multidim_evid_loss(pred,E_ref,Q_ref,device='cpu'):
    n = int(np.rint(-3.0 / 2.0 + np.sqrt(9.0 / 4.0 + 2.0 * (pred.shape[1] - 1.0))))
    nu_idx = (n * (n + 3)) // 2

    mu = pred[:, :n]

    idx = torch.tril_indices(n, n)
    L = torch.zeros(pred.shape[0], n, n, dtype=pred.dtype,device=device)
    L[:, idx[0], idx[1]] = pred[:, n:nu_idx]
    sigma = torch.matmul(L, L.transpose(1, 2))

    nu = pred[:, nu_idx]

    k = 1.0 + nu

    new_y = torch.stack([E_ref,Q_ref],dim=1)

    d = new_y - mu

    ddT_over_k = torch.matmul(d.unsqueeze(2), d.unsqueeze(1)) / k.unsqueeze(1).unsqueeze(2)

    if n == 2:
        nrm = -torch.log(nu - 1)
    else:
        nrm = torch.lgamma((nu - n + 1.0) / 2.0) - torch.lgamma((nu + 1.0) / 2.0)

    loss = ((nrm + n / 2.0 * torch.log(k)
            - nu * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=1))
            + (nu + 1) / 2.0 * torch.logdet(sigma + ddT_over_k))

    return loss.mean()

def multidim_evid_loss_all(pred,D_pred,F_pred,E_ref,Q_ref,F_ref,D_ref,wf=52.9177,wd=27.2113,device='cpu'):

    md_loss = multidim_evid_loss(pred,E_ref,Q_ref,device=device)

    # Calculate error for the forces:
    L = nn.L1Loss(reduction="mean")
    L_F = L(F_ref,F_pred)

    # Calculate error for the dipole moments:
    LD = L(D_ref,D_pred)

    total_loss = md_loss + wf*L_F + wd*LD
    return total_loss

# ------------------------------------------------------------------------------
# Define training step
# ------------------------------------------------------------------------------
def get_indices(Nref,device='cpu'):
    # Get indices pointing to batch image
    # For some reason torch does not make repetition for float
    batch_seg = torch.arange(0, Nref.size()[0],device=device).repeat_interleave(Nref.type(torch.int64))
    # Initiate auxiliary parameter
    Nref_tot = torch.tensor(0, dtype=torch.int32).to(device)

    # Indices pointing to atom at each batch image
    idx = torch.arange(end=Nref[0], dtype=torch.int32).to(device)
    # Indices for atom pairs ij - Atom i
    Ntmp = Nref.cpu()
    idx_i = idx.repeat(int(Ntmp.numpy()[0]) - 1) + Nref_tot
    # Indices for atom pairs ij - Atom j
    idx_j = torch.roll(idx, -1, dims=0) + Nref_tot
    for Na in torch.arange(2, Nref[0]):
        Na_tmp = Na.cpu()
        idx_j = torch.concat(
            [idx_j, torch.roll(idx, int(-Na_tmp.numpy()), dims=0) + Nref_tot],
            dim=0)

    # Increment auxiliary parameter
    Nref_tot = Nref_tot + Nref[0]

    # Complete indices arrays
    for Nref_a in Nref[1:]:

        rng_a = torch.arange(end=Nref_a).to(device)
        Nref_a_tmp = Nref_a.cpu()
        idx = torch.concat([idx, rng_a], axis=0)
        idx_i = torch.concat(
            [idx_i, rng_a.repeat(int(Nref_a_tmp.numpy()) - 1) + Nref_tot],
            dim=0)
        for Na in torch.arange(1, Nref_a):
            Na_tmp = Na.cpu()
            idx_j = torch.concat(
                [idx_j, torch.roll(rng_a, int(-Na_tmp.numpy()), dims=0) + Nref_tot],
                dim=0)

        # Increment auxiliary parameter
        Nref_tot = Nref_tot + Nref_a
    #Reorder the idx in i
    idx_i = torch.sort(idx_i)[0]
    # Combine indices for batch image and respective atoms
    idx = torch.stack([batch_seg, idx], dim=1)
    return idx.type(torch.int64), idx_i.type(torch.int64), idx_j.type(torch.int64), batch_seg.type(torch.int64)

def train_step(batch,num_t,loss_avg_t, emse_avg_t, emae_avg_t,
                    qmse_avg_t,qmae_avg_t,fmse_avg_t,fmae_avg_t,dmse_avg_t,dmae_avg_t,
                    pnorm,gnorm,device,maxnorm=1000):
    model.train()
    # lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=2,
    #                                                          min_lr=1e-4,verbose=True)
    batch = [i.to(device) for i in batch]
    N_t, Z_t, R_t, Eref_t, Earef_t, Fref_t, Qref_t, Qaref_t, Dref_t = batch

    # Get indices
    idx_t, idx_i_t, idx_j_t, batch_seg_t = get_indices(N_t,device=device)

    # Gather data
    Z_t = gather_nd(Z_t, idx_t)
    R_t = gather_nd(R_t, idx_t)

    if torch.count_nonzero(Earef_t) != 0:
        Earef_t = gather_nd(Earef_t, idx_t)
    if torch.count_nonzero(Fref_t) != 0:
        Fref_t = gather_nd(Fref_t, idx_t)
    if torch.count_nonzero(Qaref_t) != 0:
        Qaref_t = gather_nd(Qaref_t, idx_t)

    if DER_type == 'simple' or 'Lipz':
        energy_t, nu_t_e, alpha_t_e, beta_t_e, Qa_t,forces_t = \
            model.energy_forces_and_others_evidential(Z_t, R_t, idx_i_t, idx_j_t, Qref_t, batch_seg=batch_seg_t)

        Qtot_t = segment_sum(Qa_t, batch_seg_t, device=device)
        QR_t = torch.stack([Qa_t * R_t[:, 0], Qa_t * R_t[:, 1], Qa_t * R_t[:, 2]], 1)

        D_t = segment_sum(QR_t, batch_seg_t, device=device)

        # Calculate MAE and MSE
        # Energy
        mae_energy = torch.mean(torch.abs(energy_t - Eref_t))
        mse_energy = torch.mean(torch.square(energy_t - Eref_t))
        # Charge
        mae_charges = torch.mean(torch.abs(Qtot_t - Qaref_t))
        mse_charges = torch.mean(torch.square(Qtot_t - Qaref_t))
        # Force
        mae_forces = torch.mean(torch.abs(forces_t - Fref_t))
        mse_forces = (torch.mean(torch.square(forces_t - Fref_t)))
        # Dipole
        mae_dipole = torch.mean(torch.abs(D_t - Dref_t))
        mse_dipole = torch.mean(torch.square(D_t - Dref_t))
        # Define the loss function
        if DER_type == 'simple':
            loss_t = evid_loss_all(energy_t, nu_t_e, alpha_t_e, beta_t_e, Qtot_t,
                                   forces_t, D_t, Eref_t, Qref_t, Fref_t, Dref_t).sum()
        elif DER_type == 'Lipz':
            loss_t = lipz_loss_all(energy_t, nu_t_e, alpha_t_e, beta_t_e, Qtot_t,
                                   forces_t, D_t, Eref_t, Qref_t, Fref_t, Dref_t).sum()
    elif DER_type == 'MD':
        pred, D_t, F_t = \
            model.energy_and_forces_md_evidencial(Z_t, R_t, idx_i_t, idx_j_t, Qref_t, batch_seg=batch_seg_t)
        # Calculate MAE and MSE for different properties
        # Energy
        mae_energy = torch.mean(torch.abs(pred[:, 0] - Eref_t))
        mse_energy = torch.mean(torch.square(pred[:, 0] - Eref_t))
        # Charges
        mae_charges = torch.mean(torch.abs(pred[:, 1] - Qaref_t))
        mse_charges = torch.mean(torch.square(pred[:, 1] - Qaref_t))
        # Force
        mae_forces = torch.mean(torch.abs(F_t - Fref_t))
        mse_forces = (torch.mean(torch.square(F_t - Fref_t)))
        # Dipole
        mae_dipole = torch.mean(torch.abs(D_t - Dref_t))
        mse_dipole = torch.mean(torch.square(D_t - Dref_t))

        loss_t = multidim_evid_loss_all(pred, D_t, F_t, Eref_t, Qref_t, Fref_t, Dref_t, device=device)

    else:
        print('DER Type not recognized')
        exit()


    loss_t.backward(retain_graph=True)

    # #Gradient clip
    nn.utils.clip_grad_norm_(model.parameters(),maxnorm)
    pnorm = pnorm + compute_pnorm(model)
    gnorm = gnorm + compute_gnorm(model)
    f = num_t /(num_t + N_t.dim())
    loss_avg_t = f * loss_avg_t + (1.0 - f) * float(loss_t)
    emse_avg_t = f * emse_avg_t + (1.0 - f) * float(mse_energy)
    emae_avg_t = f * emae_avg_t + (1.0 - f) * float(mae_energy)
    fmse_avg_t = f * fmse_avg_t + (1.0 - f) * float(mse_forces)
    fmae_avg_t = f * fmae_avg_t + (1.0 - f) * float(mae_forces)
    qmse_avg_t = f * qmse_avg_t + (1.0 - f) * float(mse_charges)
    qmae_avg_t = f * qmae_avg_t + (1.0 - f) * float(mae_charges)
    dmse_avg_t = f * dmse_avg_t + (1.0 - f) * float(mse_dipole)
    dmae_avg_t = f * dmae_avg_t + (1.0 - f) * float(mae_dipole)
    num_t = num_t + N_t.dim()
    return num_t, loss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, \
           qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t, pnorm,gnorm

def valid_step(batch,num_v,loss_avg_v, emse_avg_v, emae_avg_v,
                    qmse_avg_v,qmae_avg_v,fmse_avg_v,fmae_avg_v,dmse_avg_v,dmae_avg_v,device):
    model.eval()
    batch = [i.to(device) for i in batch]
    N_v, Z_v, R_v, Eref_v, Earef_v, Fref_v, Qref_v, Qaref_v, Dref_v = batch
    # Get indices
    idx_v, idx_i_v, idx_j_v, batch_seg_v = get_indices(N_v,device=device)
    Z_v = gather_nd(Z_v, idx_v)
    R_v = gather_nd(R_v, idx_v)

    if torch.count_nonzero(Earef_v) != 0:
        Earef_v = gather_nd(Earef_v, idx_v)
    if torch.count_nonzero(Fref_v) != 0:
        Fref_v = gather_nd(Fref_v, idx_v)
    if torch.count_nonzero(Qaref_v) != 0:
        Qaref_v = gather_nd(Qaref_v, idx_v)

    if DER_type == 'simple' or 'Lipz':
        # Calculate energy, forces, and atomic properties
        energy_v, nu_v_e, alpha_v_e, beta_v_e, Qa_v, forces_v = \
            model.energy_forces_and_others_evidential(Z_v, R_v, idx_i_v, idx_j_v, Qref_v,
                                                      batch_seg=batch_seg_v)

        Qtot_v = segment_sum(Qa_v, batch_seg_v, device=device)
        QR_v = torch.stack([Qa_v * R_v[:, 0], Qa_v * R_v[:, 1], Qa_v * R_v[:, 2]], 1)

        D_v = segment_sum(QR_v, batch_seg_v, device=device)

        # Calculate MAE and MSE
        # Energy
        mae_energy = torch.mean(torch.abs(energy_v - Eref_v))
        mse_energy = torch.mean(torch.square(energy_v - Eref_v))
        # Charge
        mae_charges = torch.mean(torch.abs(Qtot_v - Qaref_v))
        mse_charges = torch.mean(torch.square(Qtot_v - Qaref_v))
        # Force
        mae_forces = torch.mean(torch.abs(forces_v - Fref_v))
        mse_forces = (torch.mean(torch.square(forces_v - Fref_v)))
        # Dipole
        mae_dipole = torch.mean(torch.abs(D_v - Dref_v))
        mse_dipole = torch.mean(torch.square(D_v - Dref_v))

        if DER_type == 'simple':
            loss_v = evid_loss_all(energy_v, nu_v_e, beta_v_e, Qtot_v,
                                   forces_v, D_v, Eref_v, Qref_v, Fref_v, Dref_v).sum()
        elif DER_type == 'Lipz':
            loss_v = lipz_loss_all(energy_v, nu_v_e, alpha_v_e, beta_v_e, Qtot_v,
                                   forces_v, D_v, Eref_v, Qref_v, Fref_v, Dref_v).sum()
        else:
            print('You should never reach this point')
            exit()

    elif DER_type == 'MD':

        pred_V, D_v, F_v = \
            model.energy_and_forces_md_evidencial(Z_v, R_v, idx_i_v, idx_j_v, Qref_v, batch_seg=batch_seg_v)

        mae_energy = torch.mean(torch.abs(pred_V[:, 0] - Eref_v))
        mse_energy = torch.mean(torch.square(pred_V[:, 0] - Eref_v))

        mae_charges = torch.mean(torch.abs(pred_V[:, 1] - Qref_v))
        mse_charges = torch.mean(torch.square(pred_V[:, 1] - Qref_v))
        # Force
        mae_forces = torch.mean(torch.abs(F_v - Fref_v))
        mse_forces = (torch.mean(torch.square(F_v - Fref_v)))
        # Dipole
        mae_dipole = torch.mean(torch.abs(D_v - Dref_v))
        mse_dipole = torch.mean(torch.square(D_v - Dref_v))

        # loss
        loss_v = multidim_evid_loss_all(pred_V, D_v, F_v, Eref_v, Qref_v, Fref_v, Dref_v, device=device)

    else:
        print('You should never reach this point')
        exit()

    f = num_v / (num_v + N_v.dim())
    loss_avg_v = f * loss_avg_v + (1.0 - f) * float(loss_v)
    emse_avg_v = f * emse_avg_v + (1.0 - f) * float(mse_energy)
    emae_avg_v = f * emae_avg_v + (1.0 - f) * float(mae_energy)
    fmse_avg_v = f * fmse_avg_v + (1.0 - f) * float(mse_forces)
    fmae_avg_v = f * fmae_avg_v + (1.0 - f) * float(mae_forces)
    qmse_avg_v = f * qmse_avg_v + (1.0 - f) * float(mse_charges)
    qmae_avg_v = f * qmae_avg_v + (1.0 - f) * float(mae_charges)
    dmse_avg_v = f * dmse_avg_v + (1.0 - f) * float(mse_dipole)
    dmae_avg_v = f * dmae_avg_v + (1.0 - f) * float(mae_dipole)
    num_v = num_v + N_v.dim()
    return num_v, loss_avg_v, emse_avg_v, emae_avg_v, fmse_avg_v, fmae_avg_v, \
           qmse_avg_v, qmae_avg_v, dmse_avg_v, dmae_avg_v



# ------------------------------------------------------------------------------
# Train PhysNet model
# ------------------------------------------------------------------------------
logging.info("starting training...")

# Define Optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate,
                             weight_decay=args.l2lambda,amsgrad=True)

lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.power(args.decay_rate,1/args.decay_steps))

# Define Exponential Moving Average
ema = ExponentialMovingAverage(model.parameters(),decay=args.ema_decay)

# Initiate epoch and step counter
epoch = torch.tensor(1, requires_grad=False, dtype=torch.int64)
step = torch.tensor(1, requires_grad=False, dtype=torch.int64)

# Initiate checkpoints and load last checkpoint
latest_ckpt = load_checkpoint(checkpoint_file)
if latest_ckpt is not None:
    model.load_state_dict(latest_ckpt['model_state_dict'])
    optimizer.load_state_dict(latest_ckpt['optimizer_state_dict'])
    epoch = latest_ckpt['epoch']

# Create validation batches
valid_batches = data.get_valid_batches()

# Initialize counter for estimated time per epoch
time_train_estimation = np.nan
time_train = 0.0
best_loss = np.Inf

# Training loop
# Terminate training when maximum number of iterations is reached
while epoch <= args.max_steps:

    # Reset error averages
    num_t, loss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, \
        qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t, pnorm_t, gnorm_t = \
        reset_averages("train",device=args.device)

    # Create train batches
    train_batches, N_train_batches = data.get_train_batches()

    # Start train timer
    train_start = time()

    # Iterate over batches
    for ib, batch in enumerate(train_batches):
        optimizer.zero_grad()
        # Start batch timer
        batch_start = time()

        # Show progress bar
        if args.show_progress:
            printProgressBar(
                ib, N_train_batches, prefix="Epoch {0: 5d}".format(
                    epoch.numpy()),
                suffix=("Complete - Remaining Epoch Time: "
                        + "{0: 4.1f} s     ".format(time_train_estimation)),
                length=42)

        # Training step

        num_t, loss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, \
        qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t, pnorm_t, gnorm_t = \
             train_step(batch,num_t, loss_avg_t, emse_avg_t, emae_avg_t, fmse_avg_t, fmae_avg_t, \
                             qmse_avg_t, qmae_avg_t, dmse_avg_t, dmae_avg_t, pnorm_t,gnorm_t,args.device)
        optimizer.step()

        ema.update()
        # Stop batch timer
        batch_end = time()

        # Actualize time estimation
        if args.show_progress:
            if ib == 0:
                time_train_estimation = (
                        (batch_end - batch_start) * (N_train_batches - 1))
            else:
                time_train_estimation = (
                        0.5 * (time_train_estimation - (batch_end - batch_start))
                        + 0.5 * (batch_end - batch_start) * (N_train_batches - ib - 1))

        # Increment step number
        step = step + 1

    # Stop train timer
    train_end = time()
    time_train = train_end - train_start

    # Show final progress bar and time
    if args.show_progress:
        loss_ev_t_temp = loss_avg_t.detach().cpu()
        lat = float(loss_ev_t_temp.numpy())
        printProgressBar(
            N_train_batches, N_train_batches, prefix="Epoch {0: 5d}".format(
                epoch.numpy()),
            suffix=("Done - Epoch Time: "
                    + "{0: 4.1f} s, Average Loss: {1: 4.4f}   ".format(
                        time_train, lat)))  # length=42))

    # Save progress
    if (epoch % args.save_interval == 0):
        number_of_ckpt = int(epoch / args.save_interval)
        save_checkpoint(model=model, epoch=epoch, name_of_ckpt=number_of_ckpt)


    # Check performance on the validation set

    if (epoch % args.validation_interval) == 0:
        # Update training results
        results_t = {}
        loss_ev_t_temp = loss_avg_t.detach().cpu()
        results_t["loss_train"] = loss_ev_t_temp.numpy()
        results_t["time_train"] = time_train
        results_t["norm_parm"] = pnorm_t
        results_t["norm_grad"] = gnorm_t
        if data.include_E:
            emae_t_temp = emae_avg_t.detach().cpu()
            emse_t_temp = emse_avg_t.detach().cpu()
            results_t["energy_mae_train"] = emae_t_temp.numpy()
            results_t["energy_rmse_train"] = np.sqrt(emse_t_temp.numpy())
        if data.include_F:
            fmae_t_temp = fmae_avg_t.detach().cpu()
            fmse_t_temp = fmse_avg_t.detach().cpu()
            results_t["force_mae_train"] = fmae_t_temp.numpy()
            results_t["force_rmse_train"] = np.sqrt(fmse_t_temp.numpy())
        if data.include_Q:
            qmae_avg_t_temp = qmae_avg_t.detach().cpu().numpy()
            qmse_avg_t_temp = qmse_avg_t.detach().cpu().numpy()
            results_t["charge_mae_train"] = qmae_avg_t_temp
            results_t["charge_rmse_train"] = np.sqrt(qmse_avg_t_temp)
        if data.include_D:
            dmae_avg_t_temp = dmae_avg_t.detach().cpu().numpy()
            dmse_avg_t_temp = dmse_avg_t.detach().cpu().numpy()
            results_t["dipole_mae_train"] = dmae_avg_t_temp
            results_t["dipole_rmse_train"] = np.sqrt(dmse_avg_t_temp)
        # Write Results to tensorboard
        for key, value in results_t.items():
            summary_writer.add_scalar(key, value, global_step=epoch)

        # # Backup variables and assign EMA variables
        # backup_vars = [tf.identity(var) for var in model.trainable_variables]
        # for var in model.trainable_variables:
        #     var.assign(ema.average(var))

        # Reset error averages
        num_v, loss_avg_v, emse_avg_v, emae_avg_v, fmse_avg_v, fmae_avg_v, \
            qmse_avg_v, qmae_avg_v, dmse_avg_v, dmae_avg_v = reset_averages('valid',device=args.device)
        # Start valid timer
        valid_start = time()

        for ib, batch in enumerate(valid_batches):
            num_v, loss_avg_v, emse_avg_v, emae_avg_v, fmse_avg_v, fmae_avg_v, \
                qmse_avg_v, qmae_avg_v, dmse_avg_v, dmae_avg_v =\
                valid_step(batch, num_v, loss_avg_v, emse_avg_v, emae_avg_v, fmse_avg_v, fmae_avg_v,
                                qmse_avg_v, qmae_avg_v, dmse_avg_v, dmae_avg_v,device=args.device)

        # Stop valid timer
        valid_end = time()
        time_valid = valid_end - valid_start

        # Update validation results
        results_v = {}
        loss_avg_v_temp = loss_avg_v.detach().cpu()
        results_v["loss_valid"] = loss_avg_v_temp.numpy()
        results_v["time_valid"] = time_valid
        if data.include_E:
            emae_v_temp = emae_avg_v.detach().cpu()
            emse_v_temp = emse_avg_v.detach().cpu()
            results_v["energy_mae_valid"] = emae_v_temp.numpy()
            results_v["energy_rmse_valid"] = np.sqrt(emse_v_temp)
        if data.include_F:
            fmae_v_temp = fmae_avg_v.detach().cpu()
            fmse_v_temp = fmse_avg_v.detach().cpu()
            results_v["force_mae_valid"] = fmae_v_temp.numpy()
            results_v["force_rmse_valid"] = np.sqrt(fmse_v_temp)
        if data.include_Q:
            qmae_avg_v_temp = qmae_avg_v.detach().cpu().numpy()
            qmse_avg_v_temp = qmse_avg_v.detach().cpu().numpy()
            results_v["charge_mae_valid"] = qmae_avg_v_temp
            results_v["charge_rmse_valid"] = np.sqrt(qmse_avg_v_temp)
        if data.include_D:
            dmae_avg_v_temp = dmae_avg_v.detach().cpu().numpy()
            dmse_avg_v_temp = dmse_avg_v.detach().cpu().numpy()
            results_v["dipole_mae_valid"] = dmae_avg_v_temp
            results_v["dipole_rmse_valid"] = np.sqrt(dmse_avg_v_temp)


        for key, value in results_v.items():
            summary_writer.add_scalar(key, value, global_step=epoch)

        if results_v["loss_valid"] < best_loss:

            # Assign results of best validation
            best_loss = results_v["loss_valid"]
            if data.include_E:
                best_emae = results_v["energy_mae_valid"]
                best_ermse = results_v["energy_rmse_valid"]
            else:
                best_emae = np.Inf
                best_ermse = np.Inf

            if data.include_F and data.include_E:
                best_fmae = results_v["force_mae_valid"]
                best_frmse = results_v["force_rmse_valid"]
            else:
                best_frmse = np.Inf
                best_fmae = np.Inf
            if data.include_Q:
                best_qmae = results_v["charge_mae_valid"]
                best_qrmse = results_v["charge_rmse_valid"]
            else:
                best_qmae = np.Inf
                best_qrmse = np.Inf
            if data.include_D:
                best_dmae = results_v["dipole_mae_valid"]
                best_drmse = results_v["dipole_rmse_valid"]
            else:
                best_dmae = np.Inf
                best_drmse = np.Inf

            best_epoch = epoch.numpy()

            # Save best results
            np.savez(
                best_loss_file, loss=best_loss,
                emae=best_emae, ermse=best_ermse,
                fmae=best_fmae, frmse=best_frmse,
                epoch=best_epoch)

            # Save best model variables
            save_checkpoint(model=model, epoch=epoch, best=True)

            # Update best results
            results_b = {}
            results_b["loss_best"] = best_loss
            if data.include_E:
                results_b["energy_mae_best"] = best_emae
                results_b["energy_rmse_best"] = best_ermse
            if data.include_F:
                results_b["force_mae_best"] = best_fmae
                results_b["force_rmse_best"] = best_frmse
            if data.include_Q:
                results_b["charge_mae_best"] = best_qmae
                results_b["charge_rmse_best"] = best_qrmse
            if data.include_D:
                results_b["dipole_mae_best"] = best_dmae
                results_b["dipole_rmse_best"] = best_drmse

             # Write the results to tensorboard
            for key, value in results_b.items():
                summary_writer.add_scalar(key, value, global_step=epoch)


        # for var, bck in zip(model.trainable_variables, backup_vars):
        #     var.assign(bck)

    #Generate summaries
    if ((epoch % args.summary_interval == 0)
            and (epoch >= args.validation_interval)):

        if data.include_E:
            print(
                "Summary Epoch: " + \
                str(epoch.numpy()) + '/' + str(args.max_steps),
                "\n    Loss   train/valid: {0: 1.3e}/{1: 1.3e}, ".format(
                    results_t["loss_train"],
                    results_v["loss_valid"]),
                " Best valid loss:   {0: 1.3e}, ".format(
                    results_b["loss_best"]),
                "\n    MAE(E) train/valid: {0: 1.3e}/{1: 1.3e}, ".format(
                    results_t["energy_mae_train"],
                    results_v["energy_mae_valid"]),
                " Best valid MAE(E): {0: 1.3e}, ".format(
                    results_b["energy_mae_best"]))

    # Increment epoch number
    lr_schedule.step()
    epoch += 1




