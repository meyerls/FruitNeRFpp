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
import time

hostname = os.uname()[1]

if hostname == 'tinyx':
    synthetic_data_base_path = '/home/woody/iwi9/iwi9019h/data/synthetic'
    synthetic_data_output_base_path = '/home/woody/iwi9/iwi9019h/data/synthetic_results/'
elif hostname == 'austro' or hostname == 'rockabilly':
    #synthetic_data_base_path = '/home/se86kimy/Dropbox/07_data/CF-NeRF/synthetic/'
    #synthetic_data_output_base_path = '/home/se86kimy/Dropbox/07_data/CF-NeRF/synthetic_results/'

    synthetic_data_base_path = '/home/se86kimy/Dropbox/07_data/CF-NeRF/synthetic'
    synthetic_data_output_base_path = '/home/se86kimy/Dropbox/05_productive/04_paper/06_CF-NeRF/temperature_cluster_sweep'
else:
    raise ValueError('Unknown hostname')

fruit_types_all = ['01_apple_tree_2048x2048_#300_GT',
                   #'01_apple_tree_2048x2048_#300_DETIC',
                   #'01_apple_tree_2048x2048_#300_SAM',
                   #'02_pear_tree_2048x2048_#300_GT',
                   #'02_pear_tree_2048x2048_#300_DETIC',
                   #'02_pear_tree_2048x2048_#300_SAM',
                   '03_plum_tree_2048x2048_#300_GT',
                   # '03_plum_tree_2048x2048_#300_DETIC',
                   # '03_plum_tree_2048x2048_#300_SAM',
                   # '05_lemon_tree_2048x2048_#300_GT',
                   # '05_lemon_tree_2048x2048_#300_DETIC',
                   # '05_lemon_tree_2048x2048_#300_SAM',
                   # '07_peach_tree_2048x2048_#300_GT',
                   # '07_peach_tree_2048x2048_#300_DETIC',
                   # '07_peach_tree_2048x2048_#300_SAM',
                   '08_mango_tree_2048x2048_#300_GT',
                   # '08_mango_tree_2048x2048_#300_DETIC',
                   # '08_mango_tree_2048x2048_#300_SAM',
                   # "FUJI_FruitNeRF_SAM"
                   ]

# with open(log_file) as f:
#    s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
#    if s.find(b'timestamp=') != -1:
#        location = s.find(b'timestamp=')
#        timestamp_line = s[location:location + 40].split(b'\n')[0]
#        timestamp = timestamp_line.split(b'\'')[1].decode("utf-8")
#    else:
#        timestamps_all = glob.glob(os.path.join(fruit_type_output_folder, nerf_type, '*'))
#        timestamps_all.sort(key=os.path.getmtime)
#        # Choose latest one
#        timestamp = os.path.basename(timestamps_all[-1])
#
# config_path = os.path.join(fruit_type_output_folder, nerf_type, timestamp, 'config.yml')
# export_path = fruit_type_output_folder

nerf_type = "cf-nerf-synthetic-cluster"

steps = 11
max_lambda = 10
lambda_cosine = 1
fixed_lambda  = "lambda_c"
variable_lambda = "lambda_e "
x_axis = np.linspace(0, max_lambda, steps)
#x_axis = np.concatenate([np.asarray([0]), np.logspace(-1, max_lambda, num=steps)])

#steps = steps + 1

results_collection = {}

for fruit_type in fruit_types_all:
    fruit_type_output_folder = Path(synthetic_data_output_base_path) / fruit_type
    #fruit_type_output_folder_cf_nerf = fruit_type_output_folder / nerf_type

    # Iterate over all timestamps
    list_timestamps = []
    for current_path in fruit_type_output_folder.glob("*"):
        list_timestamps.append(list(fruit_type_output_folder.glob("*"))[0].name)

    latest_timestamp =sorted(list_timestamps)[-1]
    fruit_type_output_folder_cf_nerf = fruit_type_output_folder / latest_timestamp / nerf_type

    fruit_type_folder_orig = Path(synthetic_data_base_path) / fruit_type


    if "FUJI" in fruit_type_folder_orig.__str__():
        gt_mesh_path = '/home/se86kimy/Dropbox/07_data/CF-NeRF/FUJI/FUJI_FruitNeRF/fuji_apple_lineset_aligned.ply'
    elif '_GT' in fruit_type_folder_orig.__str__():
        gt_mesh_basepath = fruit_type_folder_orig.__str__().split('_GT')[0]
        #pcd_export_path = os.path.join(fruit_type_output_folder_cf_nerf, latest_timestamp, nerf_type)
        gt_mesh_path = os.path.join(gt_mesh_basepath, 'fruits.obj')
    elif '_SAM' in fruit_type_folder_orig.__str__():
        gt_mesh_basepath = fruit_type_folder_orig.__str__().split('_SAM')[0]
        #pcd_export_path = os.path.join(fruit_type_output_folder_cf_nerf, latest_timestamp, nerf_type)
        gt_mesh_path = os.path.join(gt_mesh_basepath, 'fruits.obj')
    elif '_DETIC' in fruit_type_folder_orig.__str__():
        gt_mesh_basepath = fruit_type_folder_orig.__str__().split('_DETIC')[0]
        gt_mesh_path = os.path.join(gt_mesh_basepath, 'fruits.obj')
    else:
        raise ValueError('Unknown semantic mask type')


    pcd_export_path = fruit_type_output_folder_cf_nerf


    date = "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
    result_folder = './parameter_sweep_distance_function/{}'.format(date)
    counting_result = Path(pcd_export_path) / 'count_result.json'

    for lambda_eucl_dist in x_axis:
        CONSOLE.print(f"[bold green]:white_check_mark:  lambda_e={lambda_eucl_dist:.2f}, lambda_c={lambda_cosine:.2f}")

        # Create folder so save results
        os.makedirs(result_folder, exist_ok=True)

        log_file = os.path.join(result_folder, "nerf_output_count.txt")
        f = open(log_file, "w")

        print(pcd_export_path)

        if os.path.exists(counting_result):
            os.remove(counting_result)

        process = subprocess.Popen(['ns-count',
                                    '--load-pcd', pcd_export_path,
                                    '--output-dir', pcd_export_path,
                                    '--gt_pcd_file', gt_mesh_path,
                                    '--lambda_eucl_dist', str(lambda_eucl_dist),
                                    '--lambda_cosine', str(lambda_cosine),
                                    '--staged_max_points', str(200_000),
                                    '--clustering_variant', 'staged',
                                    '--staged-num-clusters', str(15)
                                    ],
                                   stdout=f,
                                   stderr=f)
        stdout, stderr = process.communicate()

        time.sleep(2)

        # Copy count result from nerf folder to result folder and rename it according to params
        result_file = Path(result_folder) / f"lambda_e_{lambda_eucl_dist:.2f}_lambda_cosine_{lambda_cosine:.2f}.json"
        shutil.copy(counting_result, result_file)

    result_counting_sweep_F1 = np.zeros(steps)
    result_counting_sweep_Precision = np.zeros(steps)
    result_counting_sweep_Recall = np.zeros(steps)
    result_counting_sweep_Count = np.zeros(steps)

    for idx, jsonpath in enumerate(sorted(glob.glob(result_folder + '/*.json'))):
        first, second = jsonpath.split('lambda_e')[-1].split('lambda_c')

        with open(jsonpath) as f:
            count_r = json.load(f)

        num_fruits = count_r["gt_apples"]

        result_counting_sweep_F1[idx] = count_r['F1']
        result_counting_sweep_Precision[idx] = count_r['Precision']
        result_counting_sweep_Recall[idx] = count_r['Recall']
        result_counting_sweep_Count[idx] = count_r['counted_apples']

    # plt.plot(x_axis, result_counting_sweep_F1, label='apples')
    # plt.title("F1 Score of Parameter Sweep $\lambda_e$")
    # plt.ylim(0.9, 1)
    # plt.show()

    results_collection.update({
        fruit_type: {
            "F1": result_counting_sweep_F1,
            "Precision": result_counting_sweep_Precision,
            "Recall": result_counting_sweep_Recall,
            "Count": result_counting_sweep_Count
        }
    })

    fig, axs = plt.subplots(2, 2)
    # fig.set("Overview of Parameter Sweep $\lambda_e$")
    fig.suptitle('Overview of Parameter Sweep $\{}$ for {}'.format(variable_lambda, fruit_type), fontsize=10)

    print('F1:',result_counting_sweep_F1)

    if False:
        axs[0, 0].plot(x_axis, result_counting_sweep_F1)
        axs[0, 0].set_title('F1')
        axs[0, 1].plot(x_axis, result_counting_sweep_Precision)
        axs[0, 1].set_title('Precision')
        axs[1, 0].plot(x_axis, result_counting_sweep_Recall, )
        axs[1, 0].set_title('Recall')
        axs[1, 1].plot(x_axis, result_counting_sweep_Count  / num_fruits)
        axs[1, 1].set_title('Count')

        for ax in axs.flat:
            ax.set(xlabel='x-label', ylabel='y-label')
            ax.set_ylim(0.0, 1.02)

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        plt.show()

line_styles = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted',
               (0, (3, 5, 1, 5)), (0, (5, 10)), (0, (3, 1, 1, 1)), (0, (1, 10)),
               (0, (5, 1)), (0, (5, 5)), (0, (5, 1, 5, 1)), (0, (3, 5, 1, 5)),
               (0, (3, 5, 1, 5, 1, 5)), (0, (5, 5, 1, 5))]
colors = plt.cm.get_cmap('tab20').colors

fig = plt.figure(figsize=[5, 5])
for idx, key in enumerate(results_collection.keys()):
    f1 = results_collection[key]["F1"]
    plt.plot(x_axis, f1,
             linestyle=line_styles[idx],
             color=colors[idx % len(colors)],
             label="".join(key.split("_tree_2048x2048_#300")))
    plt.ylim(0.0, 1.001)

plt.title("Overview of Parameter Sweep $\{}$, $\{}= 1$".format(variable_lambda, fixed_lambda))
plt.grid(True)
plt.xlabel("$\lambda_c$")
plt.ylabel("F1-Score")
plt.legend()
plt.show()

fig = plt.figure(figsize=[5, 5])
results_list = []
results_list_key = []
for idx, key in enumerate(results_collection.keys()):
    f1 = results_collection[key]["F1"]
    results_list.append(f1.tolist())
    new_key_name = "_".join("".join(key.split("_tree_2048x2048_#300")).split("_")[1:])
    results_list_key.append(new_key_name)

fig, ax = plt.subplots()
im = ax.imshow(results_list, cmap='viridis', extent=[-0.5, 2.5, -0.5, 2.5])
plt.yticks(ticks=np.arange(len(results_list_key)), labels=results_list_key)
# fig.colorbar(im, ax)
# ax.xlabel(x_axis)
plt.xticks(ticks=np.arange(x_axis.shape[0]), labels=x_axis.tolist())
ax.set_title("tbd")
fig.colorbar(im)
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
