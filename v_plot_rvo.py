from utils import *
from args import get_parser
import seaborn as sns
import scipy

# dataset = "SMD"
# group = "1-4"
# n_feature = 38
# started_dim = 0
# stgat_dir_base = "20230321_smd"

# dataset = "SMAP"
# group = "A-7"
# n_feature = 25
# started_dim = 1
# stgat_dir_base = "20230321_smap"

dataset = "MSL"
group = "P-10"
n_feature = 50
started_dim = 1
stgat_dir_base = "20230321_msl"

base_path = f"output/{dataset}/{group}/"
stgat_f1s,gat_f1s = [],[]
for i in range(started_dim,n_feature):
    stgat_dir = f"{stgat_dir_base}_{i}"
    stgat_file_path = base_path + stgat_dir + "/summary_file.txt"
    stgat_f1 = get_key_from_bf_result(stgat_file_path, "f1")
    stgat_f1s.append(stgat_f1)

    gat_dir = f"removed_variate_{i}"
    gat_file_path = base_path + gat_dir + "/summary.txt"
    gat_f1 = get_key_from_bf_result(gat_file_path, "f1")
    gat_f1s.append(gat_f1)

baseline_stgat_file_path = base_path + f"stgat_removed_variate_{n_feature}/summary.txt"
stgat_baseline_f1 = get_key_for_maml(baseline_stgat_file_path,"f1")
baseline_gat_file_path = base_path + f"mtad_gat_removed_variate_{n_feature}/summary.txt"
gat_baseline_f1 = get_key_from_bf_result(baseline_gat_file_path,"f1")
stgat_baseline_f1s = np.array([stgat_baseline_f1]).repeat(len(stgat_f1s))
gat_baseline_f1 = np.array([gat_baseline_f1]).repeat(len(stgat_f1s))

plt.figure()
# plt.plot(range(len(stgat_f1s)), stgat_f1s, label=f"stgat")
plt.plot(range(len(gat_f1s)), gat_f1s, label=f"gat")
plt.plot(range(len(gat_baseline_f1)), gat_baseline_f1,":", label=f"gat-full")
# plt.plot(range(len(stgat_baseline_f1s)), stgat_baseline_f1s,":", label=f"stgat-full")
plt.ylabel("F1 score")
plt.xlabel("The excluded variate")
plt.title(f"The performance change when excluding every variate on {dataset}_{group}")
plt.legend()
plt.savefig(f"analysis/rvo_curve_{dataset}_{group}.pdf")