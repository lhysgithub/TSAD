from utils import *
from args import get_parser
import seaborn as sns
import scipy

# get row data
parser = get_parser()
args = parser.parse_args()
args.dataset = "WT"
args.group = "WT23"

train_data, test_data, test_label = get_dataset_np(args)

for i in range(test_data.shape[1]):
    plt.figure()
    plt.plot(range(len(test_data)), test_data.T[i].T, label=f"{i}")
    plt.legend()
    plt.savefig(f"analysis/{args.group}_{i}")