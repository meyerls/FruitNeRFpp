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



"""
ns-train cf-nerf-fuji  --data /home/woody/iwi9/iwi9019h/data/FUJI/FUJI_FruitNeRF_SAM/ --output-dir /home/woody/iwi9/iwi9019h/data/FUJI_results --vis wandb --viewer.quit-on-train-completion True --max-num-iterations 125000 --pipeline.datamanager.train-num-rays-per-batch 8192 --pipeline.datamanager.eval-num-rays-per-batch 8192 --save-only-latest-checkpoint False --steps-per-save 5000 --pipeline.datamanager.train-num-images-to-sample-from -1 --pipeline.model.camera-optimizer.mode off && exit"""


fuji_data_base_path = '/home/woody/iwi9/iwi9019h/data/FUJI'
fuji_data_output_base_path = '/home/woody/iwi9/iwi9019h/data/FUJI_results'

saved_checkpoint_weights_fuji_sam_base = '/home/woody/iwi9/iwi9019h/data/FUJI_results/FUJI_FruitNeRF_SAM_base/cf-nerf-fuji/2024-09-04_125858/nerfstudio_models/step-000120000.ckpt'
saved_checkpoint_weights_fuji_detic_base = '/home/woody/iwi9/iwi9019h/data/FUJI_results/FUJI_FruitNeRF_SAM_base/cf-nerf-fuji/2024-09-04_125858/nerfstudio_models/step-000120000.ckpt'

data_day = datetime.datetime.today().date().__str__()

parser = argparse.ArgumentParser(prog='Run Synthetic Feature Dim Eval')
parser.add_argument('--model',
                    type=str,
                    default='FUJI_FruitNeRF_SAM',
                    help='FUJI_FruitNeRF_SAM or FUJI_FruitNeRF_DETIC')
parser.add_argument('--lambda_eucl_dist',
                    type=float,
                    default=0.1)
parser.add_argument('--lambda_cosine',
                    type=float,
                    default=1.0)
args = parser.parse_args()

fruit_types_dict = {
    'FUJI_FruitNeRF_SAM': saved_checkpoint_weights_fuji_sam_base,
    'FUJI_FruitNeRF_DETIC': saved_checkpoint_weights_fuji_detic_base,
}

model_types_all = [args.fruit_set]

saved_checkpoint_weights = fruit_types_dict[model_types_all[0]]
nerf_type = 'cf-nerf-fuji'

train_num_rays_per_batch = 2048 * 8
eval_num_rays_per_batch =  2_048
eval_num_points_per_side = 1_500

max_um_iterations = 250_000
training_steps_instance = 120_000
max_um_iterations = max_um_iterations - training_steps_instance

lambda_eucl_dist = args.lambda_eucl_dist
lambda_cosine = args.lambda_cosine

run = wandb.init(project='nerfstudio-project',
                 name="{}_{}_feature-vec-result-summary_lc_{}_le_{}".format(data_day,
                                                                            model_types_all[0],
                                                                            lambda_cosine,
                                                                            lambda_eucl_dist),
                 reinit=True)

# Feature vector dim to try out
feature_vector_dim_list = [1, 2, 4, 6, 8, 16, 32, 64, 128]

counting_table = []
f1_table = []
precision_table = []
recall_table = []

tempfolder = os.environ['TMPDIR']

for model_type in model_types_all:
    for feature_vector_dim in feature_vector_dim_list:
        fruit_type_folder_orig = Path(fuji_data_base_path) / model_type

        # Copy to tmp
        fruit_type_folder = Path(tempfolder) / model_type
        shutil.copytree(fruit_type_folder_orig, fruit_type_folder, dirs_exist_ok=True)
        CONSOLE.print("[bold green]:white_check_mark: Copied data to TMPDIR ({})!".format(fruit_type_folder))

        # Result folder
        fruit_type_output_folder = Path(fuji_data_output_base_path) / model_type
        os.makedirs(fruit_type_output_folder, exist_ok=True)

        config = {
            'fruit_type': model_type,
            'train_num_rays_per_batch': train_num_rays_per_batch,
            'eval_num_rays_per_batch': eval_num_rays_per_batch,
            'eval_num_points_per_side': eval_num_points_per_side,
            'base_path': fuji_data_base_path
        }

        log_file = os.path.join(fruit_type_output_folder, "nerf_output.txt")
        f = open(log_file, "wb")
        CONSOLE.print(f"[bold green]:white_check_mark: Training started for %s " % (model_type))

        process = subprocess.Popen([
            'ns-train',
            nerf_type,
            '--data', fruit_type_folder.__str__(),
            '--output-dir', fuji_data_output_base_path.__str__(),
            '--vis', 'wandb',
            '--pipeline.model.temperature', str(0.35),
            '--viewer.quit-on-train-completion', 'True',
            '--max-num-iterations', str(max_um_iterations),
            '--pipeline.datamanager.train-num-rays-per-batch', str(train_num_rays_per_batch),
            '--pipeline.datamanager.eval-num-rays-per-batch', str(train_num_rays_per_batch),
            '--pipeline.model.training-steps.cascaded-freezing', 'True',
            '--pipeline.model.camera-optimizer.mode', 'off',
            '--pipeline.model.output-dim-instance', str(feature_vector_dim),
            '--load-checkpoint', str(saved_checkpoint_weights),
            '--ignore-instance-weights', str(True),
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
                                    '--bounding-box-min', '-0.2', '-0.95', '-0.3',
                                    '--bounding-box-max', '0.2', '0.95', '0.45',
                                    '--num_rays_per_batch', str(eval_num_rays_per_batch),
                                    '--num_points_per_side', str(eval_num_points_per_side)],
                                   stdout=f,
                                   stderr=f)

        stdout, stderr = process.communicate()
        CONSOLE.print(f"[bold green]:white_check_mark:  Extracted point cloud. Path: %s" % pcd_export_path)
        CONSOLE.print(f"[bold green]:white_check_mark:  Counting Fruits.")

        gt_pcd_file = os.path.join('/home/woody/iwi9/iwi9019h/data/FUJI/FUJI_FruitNeRF', 'fuji_apple_lineset_aligned.ply')
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
                                    ],
                                   stdout=f,
                                   stderr=f)

        stdout, stderr = process.communicate()

        counting_results = os.path.join(pcd_path, "count_result.json")

        with open(counting_results, "r") as file:
            counting_result = json.load(file)

        name = "".join(model_type) + "_feature_dim_" + '{:03d}'.format(feature_vector_dim)

        counting_table.append([name, counting_result["counted_apples"]])
        counting_table.append([model_type, counting_result["gt_apples"]])

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
        wandb.log({"results/feature_dim": feature_vector_dim})
