import glob
import os.path
import subprocess
import mmap
from nerfstudio.utils.rich_utils import CONSOLE
import json
import argparse

parser = argparse.ArgumentParser(
    prog='CF-NeRF Evaluation')

parser.add_argument('-d', '--dataset_name', type=str, required=True, )

args = parser.parse_args()
data_basepath = args.dataset_name

fruit_type = '01_apple_tree'
resolution = 1024
nerf_type = 'cf-nerf-cluster'
num_images = 300
# train_num_rays_per_batch = 4096#int(4096*10)
eval_num_rays_per_batch = 4096*4  # 8000
eval_num_points_per_side = 1000
max_um_iterations = 100_000

image_folder_res = '{}_{}x{}'.format(fruit_type, resolution, resolution)
image_folder_res_num = '{}_{}x{}_#{}'.format(fruit_type, resolution, resolution, num_images)
image_abs_path = os.path.join(data_basepath, image_folder_res_num)

log_path = os.path.join(image_abs_path, 'logs')
if not os.path.exists(log_path):
    os.mkdir(log_path)

log_file = os.path.join(log_path, "nerf_output.txt")
print(log_file)

f = open(log_file, "wb+")
CONSOLE.print(
    f"[bold green]:white_check_mark: Training started for data %s px x %s px and %s images with %s iterations" % (
        resolution, resolution, num_images, max_um_iterations))


print(image_abs_path)

process = subprocess.Popen(['ns-train',
                            nerf_type, '--data', image_abs_path, '--output-dir', image_abs_path,
                            '--viewer.quit-on-train-completion', 'True',
                            '--pipeline.datamanager.eval-num-images-to-sample-from', '10',
                            '--vis', 'viewer+wandb',
                            '--viewer.camera-frustum-scale', '0.2'],
                            stdout=f,
                            stderr=f)
stdout, stderr = process.communicate()

CONSOLE.print("[bold green]:white_check_mark: Finished CF-NeRF training!")

CONSOLE.print(f"[bold green]:white_check_mark: sample implicit volume")
with open(log_file) as f:
    s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    if s.find(b'timestamp=') != -1:
        location = s.find(b'timestamp=')
        timestamp_line = s[location:location + 40].split(b'\n')[0]
        timestamp = timestamp_line.split(b'\'')[1].decode("utf-8")
    else:
        timestamps_all = glob.glob(os.path.join(image_abs_path, image_folder_res_num, nerf_type, '*'))
        timestamps_all.sort(key=os.path.getmtime)
        # Choose latest one
        timestamp = os.path.basename(timestamps_all[-1])

config_path = os.path.join(image_abs_path, image_folder_res_num, nerf_type, timestamp, 'config.yml')
export_path = image_abs_path

log_file = os.path.join(log_path, "export_output.txt")
f = open(log_file, "w")
process = subprocess.Popen(['ns-export-semantics', 'instance-pointcloud',
                            '--load-config', config_path,
                            '--output-dir', export_path,
                            '--use-bounding-box', 'True',
                            '--bounding-box-min', '-1.', '-1.', '0.01',
                            '--bounding-box-max', '1.', '1.', '1',
                            '--num_rays_per_batch', str(eval_num_rays_per_batch),
                            '--num_points_per_side', str(eval_num_points_per_side)],
                           stdout=f,
                           stderr=f)

stdout, stderr = process.communicate()
CONSOLE.print(f"[bold green]:white_check_mark:  Extracted point cloud. Path: %s" % export_path)
CONSOLE.print(f"[bold green]:white_check_mark:  Counting Fruits.")

pcd_path = os.path.join(export_path, nerf_type)
gt_mesh_file = os.path.join(image_abs_path, 'fruits.obj')

log_file = os.path.join(log_path, "count_output.txt")
f = open(log_file, "w")
process = subprocess.Popen(['ns-count',
                            '--load-pcd', pcd_path,
                            '--output-dir', export_path,
                            '--gt_pcd_file', gt_mesh_file
                            ],
                           stdout=f,
                           stderr=f)

stdout, stderr = process.communicate()
CONSOLE.print(f"[bold green]:white_check_mark:  Counted Fruits. {-1}/{-1}")
