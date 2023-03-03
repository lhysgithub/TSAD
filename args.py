import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()

    # -- Data params ---
    parser.add_argument("--dataset", type=str.upper, default="SMD")
    parser.add_argument("--group", type=str, default="3-2", help="Required for SMD dataset. <group_index>-<index>")
    parser.add_argument('--n_features', type=int, help='n_features', default=38)
    parser.add_argument('--open_maml', type=str2bool, default=False)
    parser.add_argument('--norm_model', type=str, default="norm")
    parser.add_argument('--confidence', type=int, default=5)
    parser.add_argument("--using_labeled_val", type=str2bool, default=True)
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--spec_res", type=str2bool, default=False)
    parser.add_argument("--save_dir", type=str, default="temp")

    # stgat
    parser.add_argument('--slide_win', help='slide_win', type=int, default=100)
    parser.add_argument('--slide_stride', help='slide_stride', type=int, default=1)
    parser.add_argument('--pre_term', help='pre_term', type=int, default=1)
    # STGAT layers
    parser.add_argument("--layer_numb", type=int, default=2)
    # LSTM layer
    parser.add_argument("--lstm_n_layers", type=int, default=2)
    parser.add_argument("--lstm_hid_dim", type=int, default=64)
    parser.add_argument('--batch', help='batch size', type=int, default=32)
    parser.add_argument('--lr', help='learing rate', type=int, default=1e-3)

    # -- Model params ---
    # 1D conv layer
    parser.add_argument("--kernel_size", type=int, default=7)
    # GAT layers
    parser.add_argument("--use_gatv2", type=str2bool, default=False)
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None)
    parser.add_argument("--time_gat_embed_dim", type=int, default=None)
    # GRU layer
    parser.add_argument("--gru_n_layers", type=int, default=1)
    parser.add_argument("--gru_hid_dim", type=int, default=150)
    # Forecasting Model
    parser.add_argument("--fc_n_layers", type=int, default=3)
    parser.add_argument("--fc_hid_dim", type=int, default=150)
    # Reconstruction Model
    parser.add_argument("--recon_n_layers", type=int, default=1)
    parser.add_argument("--recon_hid_dim", type=int, default=150)
    # Other
    parser.add_argument("--alpha", type=float, default=0.2)

    # --- Train params ---
    parser.add_argument("--retrain", type=str2bool, default=False)
    parser.add_argument('--epoch', help='train epoch', type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--cuda_device", type=str, default="0")
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--log_tensorboard", type=str2bool, default=True)

    # --- Predictor params ---
    parser.add_argument("--scale_scores", type=str2bool, default=False)
    parser.add_argument("--use_mov_av", type=str2bool, default=False)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--level", type=float, default=None)
    parser.add_argument("--q", type=float, default=None)
    parser.add_argument("--dynamic_pot", type=str2bool, default=False)

    # --- Other ---
    parser.add_argument("--comment", type=str, default="")

    # --- meta ---
    parser.add_argument('--n_way', type=int, help='n way', default=5)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    parser.add_argument('--seed', type=int, help='random seed', default=0)




    return parser
