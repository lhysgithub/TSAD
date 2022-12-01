from train_max_selection import *
import pygad

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

def train_and_compute_f1(args, x_val, y_val, x_train):
    x_train = torch.from_numpy(x_train).float()
    x_val = torch.from_numpy(x_val).float()
    if args.dataset == 'SMD':
        output_path = f'output/SMD/{args.group}'
    else:
        output_path = f'output/{args.dataset}'
    save_path = f"{output_path}/GA_rval"
    log_dir = f'{output_path}/logs'
    n_features = x_train.shape[1]
    args_summary = str(args.__dict__)
    target_dims = get_target_dims(args.dataset)
    if target_dims is None:
        out_dim = n_features
    elif type(target_dims) == int:
        out_dim = 1
    else:
        out_dim = len(target_dims)
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()
    model = MTAD_GAT(n_features,
                     args.lookback,
                     out_dim,
                     kernel_size=args.kernel_size,
                     use_gatv2=args.use_gatv2,
                     feat_gat_embed_dim=args.feat_gat_embed_dim,
                     time_gat_embed_dim=args.time_gat_embed_dim,
                     gru_n_layers=args.gru_n_layers,
                     gru_hid_dim=args.gru_hid_dim,
                     forecast_n_layers=args.fc_n_layers,
                     forecast_hid_dim=args.fc_hid_dim,
                     recon_n_layers=args.recon_n_layers,
                     recon_hid_dim=args.recon_hid_dim,
                     dropout=args.dropout,
                     alpha=args.alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    trainer = Trainer(model,
                      optimizer,
                      args.lookback,
                      n_features,
                      target_dims,
                      1,
                      args.bs,
                      args.init_lr,
                      forecast_criterion,
                      recon_criterion,
                      args.use_cuda,
                      save_path,
                      log_dir,
                      args.print_every,
                      args.log_tensorboard,
                      args_summary)
    train_dataset = SlidingWindowDataset(x_train, args.lookback, target_dims)
    test_dataset = SlidingWindowDataset(x_val, args.lookback, target_dims)
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, args.bs, args.val_split,
                                                                args.shuffle_dataset, test_dataset=test_dataset)
    normal_loss = trainer.fit(train_loader, val_loader)

    # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001)
    }
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1}
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    reg_level = reg_level_dict[key]

    prediction_args = {
        'dataset': args.dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": save_path,
    }
    predictor = Predictor(trainer.model, args.lookback, n_features, prediction_args)

    label = y_val[args.lookback:] if y_val is not None else None
    f1 = predictor.predict_anomalies(x_train, x_val, label)
    return f1, normal_loss, trainer.model


class GA_Input:
    def __init__(self, args=None, x_val=None, y_val=None, x_train=None, x_test=None, best_select=None, best_fitness=-1,
                 best_fitness_f1=0, best_model=None, generation=0):
        self.args = args
        self.x_val = x_val
        self.y_val = y_val
        self.x_train = x_train
        self.x_test = x_test
        self.best_select = best_select
        self.best_fitness = best_fitness
        self.best_fitness_f1 = best_fitness_f1
        self.best_model = best_model
        self.generation = generation


ga_input = GA_Input()


def fitness_func(solution, solution_idx):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    global ga_input
    select = solution.tolist()
    x_train = filter_input_by_bool(select, ga_input.x_train)
    x_test = filter_input_by_bool(select, ga_input.x_test)
    x_val = filter_input_by_bool(select, ga_input.x_val)
    f1, normal_loss, model = train_and_compute_f1(ga_input.args, x_val, ga_input.y_val, x_train)
    # fitness = 1.0 / (np.abs(f1 - 1)+ 0.000001)
    # fitness = f1 - np.sum(solution) / len(solution)
    # fitness = f1
    if f1 != f1 or f1 == None:
        f1 = 0
    if normal_loss != normal_loss:
        fitness = - 10000
    else:
        # fitness = f1 - normal_loss*5 #+ np.sum(solution) / len(solution)
        # fitness = f1# - normal_loss * 5  # + np.sum(solution) / len(solution)
        fitness = f1 + np.sum(solution) / len(solution)
    if ga_input.best_fitness < fitness:
        ga_input.best_select = select
        ga_input.best_fitness = fitness
        ga_input.best_fitness_f1 = f1
        ga_input.best_model = model
    print(
        f"solution_idx: {solution_idx} solution: {solution.tolist()} selected dims: {np.sum(select)} fitness: {fitness}")
    return fitness


def callback_generation(ga_instance):
    global ga_input
    ga_input.generation += 1
    print(
        f"Generation: {ga_instance.generations_completed} Best Solution: {ga_input.best_select} Best Fitness: {ga_input.best_fitness} Selected dims: {np.sum(ga_input.best_select)} f1: {ga_input.best_fitness_f1}")


def mutation_func(ga_instance,offspring):
    global ga_input
    for chromosome_idx in range(offspring.shape[0]):
        offspring[chromosome_idx, ga_input.generation] = random.choice(ga_instance.gene_space)
        # temp = ga_instance.gene_space
        # temp2 = [offspring[chromosome_idx, ga_input.generation]]
        # temp3 = [x for x in temp if x not in temp2]
        # offspring[chromosome_idx, ga_input.generation] = random.choice(temp3)
        # random_gene_idx = np.random.choice(range(offspring.shape[1]))
        # offspring[chromosome_idx, random_gene_idx] += np.random.choice(ga_instance.gene_space)
    return offspring


def ga_selection(args, x_val, y_val, x_train, x_test):
    feature_numbers = x_train.shape[1]
    global ga_input
    ga_input = GA_Input(args, x_val, y_val, x_train, x_test)
    ga_instance = pygad.GA(num_generations=feature_numbers,
                           num_parents_mating=2,
                           fitness_func=fitness_func,
                           sol_per_pop=4,
                           num_genes=feature_numbers,
                           on_generation=callback_generation,
                           parent_selection_type="rank",
                           crossover_type="two_points",
                           # mutation_type="random",
                           # on_parents=pygad.GA.rank_selection,
                           # on_crossover=pygad.GA.two_points_crossover,
                           on_mutation=mutation_func,
                           # mutation_percent_genes=0.1,
                           gene_space=[0, 1],
                           save_solutions=True)
    ga_instance.run()
    # ga_instance.plot_fitness()
    # ga_instance.plot_result()
    # ga_instance.plot_genes()
    # ga_instance.plot_new_solution_rate()
    # solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # x = np.tanh(solution)
    # select = np.greater(x, 0)
    select = ga_input.best_select
    # select = solution.tolist()
    x_train = filter_input_by_bool(select, ga_input.x_train)
    x_test = filter_input_by_bool(select, ga_input.x_test)
    x_val = filter_input_by_bool(select, ga_input.x_val)
    return x_train, x_test, select


def main():
    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    parser = get_parser()
    args = parser.parse_args()

    dataset = args.dataset
    window_size = args.lookback
    spec_res = args.spec_res
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    group_index = args.group[0]
    index = args.group[2:]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    args_summary = str(args.__dict__)
    print(args_summary)

    if dataset == 'SMD':
        output_path = f'output/SMD/{args.group}'
        # (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
        (x_train, _), (x_test, y_test) = get_data_from_source(args, normalize=normalize)
    elif dataset in ['MSL', 'SMAP']:
        output_path = f'output/{dataset}/{args.group}'
        # (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
        (x_train, _), (x_test, y_test) = get_data_from_source(args, normalize=normalize)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')
    args.feature_numbers = x_train.shape[1]

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    x_val, y_val, x_test, y_test = split_val_set(x_test, y_test, args.val_split)
    x_train, x_test, selected = ga_selection(args, x_val, y_val, x_train, x_test)
    args.select = selected
    save_path = f"{output_path}/GA"
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]

    target_dims = get_target_dims(dataset)
    if target_dims is None:
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )

    model = ga_input.best_model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        recon_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )

    args_summary = str(args.__dict__)
    print(args_summary)

    trainer.fit(train_loader, val_loader)

    plot_losses(trainer.losses, save_path=save_path, plot=False)

    # Check test loss
    test_loss = trainer.evaluate(test_loader)
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test reconstruction loss: {test_loss[1]:.5f}")
    print(f"Test total loss: {test_loss[2]:.5f}")

    # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001)
    }
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1}
    key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    reg_level = reg_level_dict[key]

    trainer.load(f"{save_path}/model.pt")
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": save_path,
    }
    best_model = trainer.model
    predictor = Predictor(
        best_model,
        window_size,
        n_features,
        prediction_args,
    )

    label = y_test[window_size:] if y_test is not None else None
    predictor.predict_anomalies(x_train, x_test, label)

    # Save config
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)

    file_name = save_path + "/" + "summary.txt"
    return get_f1(file_name)


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    main()
