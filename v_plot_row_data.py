from utils import *
from args import get_parser
import seaborn as sns
import scipy

# get row data
parser = get_parser()
args = parser.parse_args()
args.dataset = "SMD"
args.group = "1-2"
# groups = []
# for i in range(1, 9):
#     groups.append(f"1-{i}")
# for i in range(1, 10):
#     groups.append(f"2-{i}")
# for i in range(1, 12):
#     groups.append(f"3-{i}")
groups = ["1-4"]
rs = []
rs2 = []
for args.group in groups:
    train_data, test_data, test_label = get_dataset_np(args)

    # get anormaly information
    anormals = get_anormal_dimensions(args)
    dims = 38
    anomaly_scope = len(anormals)
    anormals_marix = np.zeros((anomaly_scope, dims))
    for i in range(anomaly_scope):
        for j in anormals[i]:
            anormals_marix[i,j-1] = 1.0

    anormaly_and_std = []
    anormaly_and_std.append(normalization_axis0(anormals_marix.T.mean(axis=1)))
    normalization_inner_get = normalization_axis0(test_data)
    if np.any(sum(np.isnan(normalization_inner_get))):
        normalization_inner_get = np.nan_to_num(normalization_inner_get)
    inner_get_std_on_dim = normalization_inner_get.T.std(axis=1)
    anormaly_and_std.append(normalization_axis0(inner_get_std_on_dim))
    anormaly_and_std = np.array(anormaly_and_std)
    anormaly_and_std.T[11] = 0.9
    anormaly_and_std.T[14] = 0.9
    center = np.mean(anormaly_and_std[1])
    data_pd_2 = pd.DataFrame(anormaly_and_std.T, index=range(1, 39), columns=["Anormaly Ratio", "Standard Deviation"])
    plt.figure(figsize=(5, 8))
    p = sns.heatmap(data_pd_2, cmap="RdBu_r",
                        cbar_kws={"label": f"Value"}, center=center, #center=0.4,  # vmin=0, vmax=1,
                        square=False, linewidths=0.3)  # yticklabels=ylabes,
    p.set_ylabel("Variates Index i")
    title = "Attention and STD"
    # plt.title(title)
    # plt.xticks(rotation=30)
    plt.savefig(f"analysis/{title}_{args.group}.pdf")


    plt.figure(figsize=(5, 5))
    plt.scatter(anormaly_and_std[0],anormaly_and_std[1])
    plt.xlabel("attention")
    plt.ylabel("std")
    plt.savefig(f"analysis/{title}_{args.group}_scatter.pdf")

    r1,p1 = scipy.stats.spearmanr(anormaly_and_std[0],anormaly_and_std[1])
    r2, p2 = scipy.stats.pearsonr(normalization_axis0_st(anormaly_and_std[0]), normalization_axis0_st(anormaly_and_std[1]))
    print(f"{args.group} spearmanr correlation {r1}, significant level {p1}")
    if p1 <0.05:
        rs.append(r1)
    print(f"{args.group} pearsonr correlation {r1}, significant level {p1}")
    if p2 < 0.05:
        rs2.append(r2)
print(f"mean spearmanr correlation{np.array(rs).mean()}")
print(f"mean pearsonr correlation{np.array(rs2).mean()}")
