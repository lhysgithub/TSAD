from args import get_parser
import json
import numpy as np

parser = get_parser()
args = parser.parse_args()
args.dataset = "ZX"
args.group = "computer-b0503-03"
args.save_dir = "temp"
args.condition_control = True
save_path = f"output/{args.dataset}/{args.group}/{args.save_dir}"
index = []
scope_mses =[]
for args.slide_win in [24,48,96,288]:
        for args.epoch in [1,2,4,6,8]:
            file_name = f"results_scope_{args.condition_control}_{args.slide_win}_{args.epoch}_origin.json"
            with open(f'{save_path}/{file_name}') as f:
                result = json.load(f)
            scope_mses.append(result["point_mse"]) # point_mse scope_mse
            index.append(f"{args.slide_win}_{args.epoch}")
            
            
scope_mses_np = np.array(scope_mses)
index_np = np.array(index)
sort_index = np.argsort(scope_mses_np)
print(index_np[sort_index])
result_mse = scope_mses_np[sort_index]
for i in result_mse:
    print("%.3f"%(i))
