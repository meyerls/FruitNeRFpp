import glob
import os.path
import subprocess
import mmap
from nerfstudio.utils.rich_utils import CONSOLE
# from clustering.clustering import AppleClustering
# from clustering.clustering_apple import AppleClustering
import json
from pathlib import Path
import numpy as np
import shutil
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import datetime

# pcd_export_path = '/home/se86kimy/Dropbox/05_productive/01_code/15_ContrastiveFruitNeRF/CF-NeRF/debug/outputs/01_apple_tree_2048x2048_#300_GT/cf-nerf/2024-08-08_084142/cf-nerf'
pcd_export_path = '/home/se86kimy/Dropbox/07_data/CF-NeRF/synthetic_results/01_apple_tree_2048x2048_#300_GT/cf-nerf-cluster/2024-08-21_162812/cf-nerf-cluster'
gt_mesh_path = '/home/se86kimy/Dropbox/07_data/CF-NeRF/synthetic/01_apple_tree_2048x2048_#300/fruits.obj'

date = "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
result_folder = './parameter_sweep_distance_function/{}'.format(date)
counting_result = Path(pcd_export_path) / 'count_result.json'

lambda_e = 1

steps = 10
max_lambda_c = 4
x_axis = np.linspace(0, max_lambda_c, steps)

for lambda_c in x_axis:
    CONSOLE.print(f"[bold green]:white_check_mark:  lambda_e={lambda_e:.2f}, lambda_c={lambda_c:.2f}")

    # Create folder so save results
    os.makedirs(result_folder, exist_ok=True)

    log_file = os.path.join(result_folder, "nerf_output_count.txt")
    f = open(log_file, "w")
    process = subprocess.Popen(['ns-count',
                                '--load-pcd', pcd_export_path,
                                '--output-dir', pcd_export_path,
                                '--gt_pcd_file', gt_mesh_path,
                                '--lambda_eucl_dist', str(lambda_e),
                                '--lambda_cosine', str(lambda_c),
                                ],
                               stdout=f,
                               stderr=f)
    stdout, stderr = process.communicate()

    # Copy count result from nerf folder to result folder and rename it according to params
    result_file = Path(result_folder) / f"lambda_e_{lambda_e:.2f}_lambda_cosine_{lambda_c:.2f}.json"
    shutil.copy(counting_result, result_file)

result_counting_sweep_F1 = np.zeros(steps)
result_counting_sweep_Precision = np.zeros(steps)
result_counting_sweep_Recall = np.zeros(steps)
result_counting_sweep_Count = np.zeros(steps)

for idx, jsonpath in enumerate(sorted(glob.glob(result_folder + '/*.json'))):
    first, second = jsonpath.split('lambda_e')[-1].split('lambda_c')
    lambda_e = first.replace('_', '')
    lambda_c = second.strip('.json').split('_')[-1]

    with open(jsonpath) as f:
        count_r = json.load(f)

    result_counting_sweep_F1[idx] = count_r['F1']
    result_counting_sweep_Precision[idx] = count_r['Precision']
    result_counting_sweep_Recall[idx] = count_r['Recall']
    result_counting_sweep_Count[idx] = count_r['counted_apples']

# plt.plot(x_axis, result_counting_sweep_F1, label='apples')
# plt.title("F1 Score of Parameter Sweep $\lambda_c$")
# plt.ylim(0, 1)
# plt.show()

fig, axs = plt.subplots(2, 2)
fig.suptitle('Overview of Parameter Sweep $\lambda_c$', fontsize=10)

axs[0, 0].plot(x_axis, result_counting_sweep_F1)
axs[0, 0].set_title('F1')
axs[0, 1].plot(x_axis, result_counting_sweep_Precision, 'tab:orange')
axs[0, 1].set_title('Precision')
axs[1, 0].plot(x_axis, result_counting_sweep_Recall, )
axs[1, 0].set_title('Recall')
axs[1, 1].plot(x_axis, result_counting_sweep_Count)
axs[1, 1].set_title('Count')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()

"""
plt.imshow(data, cmap='viridis', extent=[0, 1, 0, 1])

plt.title("Heatmap of lambda sweep")
plt.ylabel('$\lambda_c$')
plt.xlabel('$\lambda_e$')
plt.xticks(np.arange(0, 1.1, step=0.1))
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.show()

#

# plotting the heatmap with seaborn
hm = sns.heatmap(data=data,
                 annot=True)
plt.title("Heatmap of lambda sweep")
plt.ylabel('$\lambda_c$')
plt.xlabel('$\lambda_e$')
plt.show()
"""
