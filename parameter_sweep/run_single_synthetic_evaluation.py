import glob
import os.path
import subprocess
import mmap
from nerfstudio.utils.rich_utils import CONSOLE
# from clustering.clustering import AppleClustering
# from clustering.clustering_apple import AppleClustering
import json
from pathlib import Path
import argparse
import wandb
import datetime
import shutil
from os.path import commonprefix
# synthetic_data_base_path = '/home/se86kimy/Dropbox/07_data/CF-NeRF/synthetic/'
synthetic_data_base_path = '/home/woody/iwi9/iwi9019h/data/synthetic'
# synthetic_data_output_base_path = '/home/se86kimy/Dropbox/07_data/CF-NeRF/synthetic_results/'
synthetic_data_output_base_path = '/home/woody/iwi9/iwi9019h/data/synthetic_results/'

parser = argparse.ArgumentParser(prog='RunSyntheticEval')
parser.add_argument('--fruit_set', metavar='N', type=int, nargs='+')
parser.add_argument('--counting_result', type=str)

# fruit_types_all = ['01_apple_tree_2048x2048_#300_GT',
#               '01_apple_tree_2048x2048_#300_DETIC',
#               '01_apple_tree_2048x2048_#300_SAM',
#               '02_pear_tree_2048x2048_#300_GT',
#               '02_pear_tree_2048x2048_#300_DETIC',
#               '02_pear_tree_2048x2048_#300_SAM',
#               '03_plum_tree_2048x2048_#300_GT',
#               '03_plum_tree_2048x2048_#300_DETIC',
#               '03_plum_tree_2048x2048_#300_SAM',
#               '05_lemon_tree_2048x2048_#300_GT',
#               '05_lemon_tree_2048x2048_#300_DETIC',
#               '05_lemon_tree_2048x2048_#300_SAM',
#               '07_peach_tree_2048x2048_#300_GT',
#               '07_peach_tree_2048x2048_#300_DETIC',
#               '07_peach_tree_2048x2048_#300_SAM',
#               '08_mango_tree_2048x2048_#300_GT',
#               '08_mango_tree_2048x2048_#300_DETIC',
#               '08_mango_tree_2048x2048_#300_SAM',
#               ]

args = parser.parse_args()

if args.fruit_set[0] == 1:
    fruit_types = [
        '01_apple_tree_2048x2048_#300_GT',
        '01_apple_tree_2048x2048_#300_DETIC',
        '01_apple_tree_2048x2048_#300_SAM',
    ]
elif args.fruit_set[0] == 2:
    fruit_types = [
        '02_pear_tree_2048x2048_#300_GT',
        '02_pear_tree_2048x2048_#300_DETIC',
        '02_pear_tree_2048x2048_#300_SAM',
    ]
elif args.fruit_set[0] == 3:
    fruit_types = [
        '03_plum_tree_2048x2048_#300_GT',
        '03_plum_tree_2048x2048_#300_DETIC',
        '03_plum_tree_2048x2048_#300_SAM',
    ]
elif args.fruit_set[0] == 4:
    fruit_types = [
        '05_lemon_tree_2048x2048_#300_GT',
        '05_lemon_tree_2048x2048_#300_DETIC',
        '05_lemon_tree_2048x2048_#300_SAM',
    ]
elif args.fruit_set[0] == 5:
    fruit_types = [
        '07_peach_tree_2048x2048_#300_GT',
        '07_peach_tree_2048x2048_#300_DETIC',
        '07_peach_tree_2048x2048_#300_SAM',
    ]
elif args.fruit_set[0] == 6:
    fruit_types = [
        '08_mango_tree_2048x2048_#300_GT',
        '08_mango_tree_2048x2048_#300_DETIC',
        '08_mango_tree_2048x2048_#300_SAM',
    ]
common_string = commonprefix(fruit_types)

nerf_type = 'cf-nerf-synthetic-cluster'

train_num_rays_per_batch = 2048 * 8
eval_num_rays_per_batch = 2_048 * 2
eval_num_points_per_side = 1_200
# max_um_iterations = 200_000

training_steps_semantic = 25_000
training_steps_instance = 35_000
max_um_iterations = 80_000

# Schedular
semantic_schedular_warmup = training_steps_semantic
semantic_schedular_max_step = (training_steps_instance + training_steps_semantic) // 2

base_field_schedular_warmup = 0
base_field_schedular_max_step = semantic_schedular_max_step

instance_field_schedular_warmup = training_steps_instance
instance_field_schedular_max_step = 0.8 * max_um_iterations if 0.8 * max_um_iterations > training_steps_instance else (training_steps_instance + max_um_iterations) / 2
instance_field_schedular_max_step = int(instance_field_schedular_max_step)

counting_table = []
f1_table = []
precision_table = []
recall_table = []

tempfolder = os.environ['TMPDIR']

lambda_eucl_dist = 1
lambda_cosine = 1

data_day = datetime.datetime.today().date().__str__()
run = wandb.init(project='nerfstudio-project', name="{}_{}_counting-result-summary".format(data_day, common_string), reinit=True)


for fruit_type in fruit_types:
    fruit_type_folder_orig = Path(synthetic_data_base_path) / fruit_type

    # Copy to tmp
    fruit_type_folder = Path(tempfolder) / fruit_type
    shutil.copytree(fruit_type_folder_orig, fruit_type_folder)
    CONSOLE.print("[bold green]:white_check_mark: Copied data to TMPDIR ({})!".format(fruit_type_folder))

    # Result folder
    fruit_type_output_folder = Path(synthetic_data_output_base_path) / fruit_type
    os.makedirs(fruit_type_output_folder, exist_ok=True)

    config = {
        'fruit_type': fruit_type,
        'train_num_rays_per_batch': train_num_rays_per_batch,
        'eval_num_rays_per_batch': eval_num_rays_per_batch,
        'eval_num_points_per_side': eval_num_points_per_side,
        'max_um_iterations': max_um_iterations,
        'base_path': synthetic_data_base_path
    }

    log_file = os.path.join(fruit_type_output_folder, "nerf_output.txt")
    f = open(log_file, "wb")
    CONSOLE.print(f"[bold green]:white_check_mark: Training started for %s " % (fruit_type))

    process = subprocess.Popen([
        'ns-train',
        nerf_type,
        '--data', fruit_type_folder.__str__(),
        '--output-dir', synthetic_data_output_base_path,
        '--vis', 'wandb',
        '--pipeline.model.temperature', str(0.2),
        '--viewer.quit-on-train-completion', 'True',
        '--max-num-iterations', str(max_um_iterations),
        '--pipeline.datamanager.train-num-rays-per-batch', str(train_num_rays_per_batch),
        '--pipeline.datamanager.eval-num-rays-per-batch', str(train_num_rays_per_batch),
        '--pipeline.model.training-steps.semantic', str(training_steps_semantic),
        '--pipeline.model.training-steps.instance', str(training_steps_instance),
        '--optimizers.base-field.scheduler.warmup-steps', str(base_field_schedular_warmup),
        '--optimizers.base-field.scheduler.max-steps', str(base_field_schedular_max_step),
        '--optimizers.semantic-field.scheduler.warmup-steps', str(semantic_schedular_warmup),
        '--optimizers.semantic-field.scheduler.max-steps', str(semantic_schedular_max_step),
        '--optimizers.instance-field.scheduler.warmup-steps', str(instance_field_schedular_warmup),
        '--optimizers.instance-field.scheduler.max-steps', str(instance_field_schedular_max_step),
        '--pipeline.model.training-steps.cascaded-freezing', 'True',
        '--pipeline.model.camera-optimizer.mode', 'off'
    ],
        stdout=f,
        stderr=f)
    stdout, stderr = process.communicate()

    CONSOLE.print("[bold green]:white_check_mark: Finished NeRF training!")

    CONSOLE.print(f"[bold green]:white_check_mark: sample implicit volume")
    with open(log_file) as f:
        s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if s.find(b'timestamp=') != -1:
            location = s.find(b'timestamp=')
            timestamp_line = s[location:location + 40].split(b'\n')[0]
            timestamp = timestamp_line.split(b'\'')[1].decode("utf-8")
        else:
            timestamps_all = glob.glob(os.path.join(fruit_type_output_folder, nerf_type, '*'))
            timestamps_all.sort(key=os.path.getmtime)
            # Choose latest one
            timestamp = os.path.basename(timestamps_all[-1])

    config_path = os.path.join(fruit_type_output_folder, nerf_type, timestamp, 'config.yml')
    export_path = fruit_type_output_folder

    pcd_export_path = os.path.join(fruit_type_output_folder, nerf_type, timestamp)

    log_file = os.path.join(fruit_type_output_folder, "nerf_output_export.txt")
    f = open(log_file, "w")
    process = subprocess.Popen(['ns-export-semantics', 'instance-pointcloud',
                                '--load-config', config_path,
                                '--output-dir', pcd_export_path,
                                '--use-bounding-box', 'True',
                                '--bounding-box-min', '-1', '-1', '-0.55',
                                '--bounding-box-max', '1', '1', '0.55',
                                '--num_rays_per_batch', str(eval_num_rays_per_batch),
                                '--num_points_per_side', str(eval_num_points_per_side)],
                               stdout=f,
                               stderr=f)

    stdout, stderr = process.communicate()
    CONSOLE.print(f"[bold green]:white_check_mark:  Extracted point cloud. Path: %s" % pcd_export_path)
    CONSOLE.print(f"[bold green]:white_check_mark:  Counting Fruits.")

    if '_GT' in fruit_type_folder.__str__():
        gt_mesh_basepath = fruit_type_folder_orig.__str__().split('_GT')[0]
    elif '_SAM' in fruit_type_folder.__str__():
        gt_mesh_basepath = fruit_type_folder_orig.__str__().split('_SAM')[0]
    elif '_DETIC' in fruit_type_folder.__str__():
        gt_mesh_basepath = fruit_type_folder_orig.__str__().split('_DETIC')[0]

    gt_pcd_file = os.path.join(gt_mesh_basepath, 'fruits.obj')
    pcd_path = os.path.join(pcd_export_path, nerf_type)

    log_file = os.path.join(fruit_type_output_folder, "nerf_output_count.txt")
    f = open(log_file, "w")
    process = subprocess.Popen(['ns-count',
                                '--load-pcd', pcd_path,
                                '--output-dir', pcd_path,
                                '--gt_pcd_file', gt_pcd_file,
                                '--lambda_eucl_dist', str(lambda_eucl_dist),
                                '--lambda_cosine', str(lambda_cosine),
                                '--staged_max_points', str(600_000),
                                '--clustering_variant', 'staged',
                                '--staged-num-clusters', str(40)                                ],
                               stdout=f,
                               stderr=f)

    stdout, stderr = process.communicate()

    counting_results = os.path.join(pcd_path, "count_result.json")

    with open(counting_results, "r") as file:
        counting_result = json.load(file)

    name = "".join(fruit_type.split("2048x2048_#300_"))

    counting_table.append([name, counting_result["counted_apples"]])
    counting_table.append([fruit_type.split("_2048x2048_#300_")[0], counting_result["gt_apples"]])

    f1_table.append([name, counting_result["F1"]])
    precision_table.append([name, counting_result["Precision"]])
    recall_table.append([name, counting_result["Recall"]])

    table_counting = wandb.Table(data=counting_table, columns=["label", "value"])
    table_f1 = wandb.Table(data=f1_table, columns=["label", "value"])
    table_precision = wandb.Table(data=precision_table, columns=["label", "value"])
    table_recall = wandb.Table(data=recall_table, columns=["label", "value"])

    wandb.log({"results/counting": wandb.plot.bar(table_counting, "label", "value", title="Counting Bar Chart")})
    wandb.log({"results/f1": wandb.plot.bar(table_f1, "label", "value", title="F1 Bar Chart")})
    wandb.log({"results/precision": wandb.plot.bar(table_precision, "label", "value", title="Precision Bar Chart")})
    wandb.log({"results/recall": wandb.plot.bar(table_recall, "label", "value", title="Recall Bar Chart")})

