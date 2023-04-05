from utils import *
from args import get_parser
import seaborn as sns
import scipy

dataset = "SMD"
group = "1-4"
n_feature = 38
started_dim = 0

# dataset = "SMAP"
# group = "A-7"
# n_feature = 25
# started_dim = 1

# dataset = "MSL"
# group = "T-13"
# n_feature = 50
# started_dim = 1

base_path = f"output/{dataset}/{group}/"
stgat_f1s,gat_f1s = [],[]
for i in range(started_dim,n_feature+1):
    temp_stgat_f1s,temp_gat_f1s = [],[]
    for j in range(4):
        stgat_dir = f"stgat_removed_variate_{i}_{j}"
        stgat_file_path = base_path + stgat_dir + "/summary.txt"
        stgat_f1 = get_key_for_maml(stgat_file_path, "f1")
        temp_stgat_f1s.append(stgat_f1)

        gat_dir = f"mtad_gat_removed_variate_{i}_{j}"
        gat_file_path = base_path + gat_dir + "/summary.txt"
        gat_f1 = get_key_from_bf_result(gat_file_path, "f1")
        temp_gat_f1s.append(gat_f1)
    gat_f1s.append(np.array(temp_gat_f1s).mean())
    stgat_f1s.append(np.array(temp_stgat_f1s).mean())

plt.figure(figsize=(8,6))
plt.plot(range(1+started_dim,len(gat_f1s[:-1])+1+started_dim), gat_f1s[:-1],"b", label=f"MTAD-GAT-w/o")
plt.plot(range(1+started_dim,len(gat_f1s[:-1])+1+started_dim), np.array([gat_f1s[-1]]).repeat(len(gat_f1s[:-1])),":b", label=f"MTAD-GAT-full")
plt.plot(range(1+started_dim,len(stgat_f1s[:-1])+1+started_dim), stgat_f1s[:-1],"r", label=f"STGAT-w/o")
plt.plot(range(1+started_dim,len(stgat_f1s[:-1])+1+started_dim), np.array([stgat_f1s[-1]]).repeat(len(stgat_f1s[:-1])),":r", label=f"STGAT-full")
plt.ylabel("F1 score")
plt.grid()
plt.xlabel("The excluded variate")
plt.title(f"The performance change when excluding variates on {dataset}_{group}")
plt.legend()
plt.savefig(f"analysis/rvo_curve_{dataset}_{group}.pdf")

stgat_base = stgat_f1s[-1]
gat_base = gat_f1s[-1]
result = np.ones_like(gat_f1s[:-1],dtype=np.int64)*0.5
for i in range(len(gat_f1s[:-1])):
    if (gat_f1s[i] <= 0.9625*gat_base and stgat_f1s[i] <= 0.9625*stgat_base):
        result[i] = 1
    elif (gat_f1s[i] >= 1.0375*gat_base and stgat_f1s[i] >= 1.0375*stgat_base):
        result[i] = 0
re = {"re":list(result)}
with open("result.txt", "w") as f:
    json.dump(re, f, indent=2)
