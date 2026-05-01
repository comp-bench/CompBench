import argparse
import os
import json
import csv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

# Metrics Imports
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from transformers import CLIPModel, CLIPProcessor

class CompBenchEvaluator:
    def __init__(self, device="cuda", clip_path="/data/jbh/clip-vit-large-patch14", metric="all"):
        self.device = device
        self.metric = metric
        print(f"Loading metrics models for [{metric}] to {device}...")
        
        # 1. LC-T (Local CLIP Text)
        if metric in ['all', 'LC-T']:
            self.clip_score_metric = CLIPScore(model_name_or_path=clip_path).to(device)
        
        # 2. LC-I (Local CLIP Image)
        if metric in ['all', 'LC-I']:
            self.clip_model = CLIPModel.from_pretrained(clip_path).to(device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
        
        # 3. Background Metrics
        if metric in ['all', 'PSNR']:
            self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        if metric in ['all', 'SSIM']:
            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        if metric in ['all', 'LPIPS']:
            self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        
        print("Models loaded successfully.")

    def preprocess_images(self, ori_path, edited_path, mask_path, gt_path=None, size=512):
        """统一的图像读取和放缩逻辑"""
        def load_and_resize(path, mode, resample):
            if path and os.path.exists(path):
                try:
                    img = Image.open(path).convert(mode)
                    return img.resize((size, size), resample=resample)
                except Exception as e:
                    return None
            return None

        ori_img = load_and_resize(ori_path, "RGB", Image.Resampling.BICUBIC)
        edited_img = load_and_resize(edited_path, "RGB", Image.Resampling.BICUBIC)
        mask_img = load_and_resize(mask_path, "L", Image.Resampling.NEAREST)
        gt_img = load_and_resize(gt_path, "RGB", Image.Resampling.BICUBIC) if gt_path else None

        return ori_img, edited_img, mask_img, gt_img

    def get_bbox_from_mask(self, mask_img):
        mask_array = np.array(mask_img)
        y, x = np.where(mask_array > 0)
        if len(y) == 0 or len(x) == 0:
            return None
        top, bottom = np.min(y), np.max(y)
        left, right = np.min(x), np.max(x)
        return (left, top, right, bottom)

    @torch.no_grad()
    def calculate_metrics(self, ori_img, edited_img, mask_img, gt_img, captions):
        """
        Input: PIL Images, captions (List of strings)
        Output: Dictionary of metrics
        """
        if ori_img is None or edited_img is None or mask_img is None:
            return None

        results = {}
        
        def to_tensor(pil_img):
            return torch.tensor(np.array(pil_img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

        ori_tensor = to_tensor(ori_img)
        edited_tensor = to_tensor(edited_img)
        
        mask_np = np.array(mask_img, dtype=np.float32) / 255.0
        mask_bg = 1 - mask_np
        mask_bg_tensor = torch.tensor(mask_bg).unsqueeze(0).unsqueeze(0).to(self.device)
        
        bg_ori = ori_tensor * mask_bg_tensor
        bg_edited = edited_tensor * mask_bg_tensor

        if self.metric in ['all', 'PSNR']:
            results['PSNR'] = self.psnr_metric(bg_edited, bg_ori).item()
        if self.metric in ['all', 'SSIM']:
            results['SSIM'] = self.ssim_metric(bg_edited, bg_ori).item()
        if self.metric in ['all', 'LPIPS']:
            results['LPIPS'] = self.lpips_metric(bg_edited * 2 - 1, bg_ori * 2 - 1).item()

        bbox = self.get_bbox_from_mask(mask_img)
        
        if bbox:
            left, top, right, bottom = bbox
            if right > left and bottom > top:
                crop_edited = edited_img.crop(bbox)
                crop_edited_tensor = torch.tensor(np.array(crop_edited)).permute(2, 0, 1).to(self.device)
                
                # LC-T 支持多 caption 取平均（主要针对 multi-object 任务）
                if self.metric in ['all', 'LC-T']:
                    lc_t_scores = []
                    for cap in captions:
                        lc_t_scores.append(self.clip_score_metric(crop_edited_tensor, cap).item())
                    results['LC-T'] = np.mean(lc_t_scores) if lc_t_scores else 0.0

                if self.metric in ['all', 'LC-I']:
                    if gt_img:
                        crop_gt = gt_img.crop(bbox)
                        inputs = self.clip_processor(images=[crop_edited, crop_gt], return_tensors="pt", padding=True).to(self.device)
                        feats = self.clip_model.get_image_features(**inputs)
                        results['LC-I'] = F.cosine_similarity(feats[0].unsqueeze(0), feats[1].unsqueeze(0)).item()
                    else:
                        results['LC-I'] = np.nan
            else:
                if self.metric in ['all', 'LC-T']: results['LC-T'] = 0.0
                if self.metric in ['all', 'LC-I']: results['LC-I'] = 0.0
        else:
            if self.metric in ['all', 'LC-T']: results['LC-T'] = 0.0
            if self.metric in ['all', 'LC-I']: results['LC-I'] = 0.0

        return results

def get_all_task_configs(data_root, model_results_root, model_name):
    tasks = [] 
    # 1. Local Editing (强制使用 metadata.json)
    for t in ['add', 'remove', 'replace']:
        tasks.append({
            'group': 'Local Editing',
            'subtask': t,
            'type': 'local',
            'info_file': f"{data_root}/{t}/metadata.json",
            'ori_dir': f"{data_root}/{t}/input_image",
            'mask_dir': f"{data_root}/{t}/mask",
            'gt_dir': f"{data_root}/{t}/edited_image",
            'res_dir': f"{model_results_root}/{model_name}_local/{t}"
        })

    # 2. Implicit Reasoning (强制使用 implicit_info.jsonl)
    tasks.append({
        'group': 'Implicit Reasoning',
        'subtask': 'implicit',
        'type': 'implicit',
        'info_file': f"{data_root}/implicit_reasoning/implicit_info.jsonl",
        'ori_dir': f"{data_root}/implicit_reasoning/ori_images",
        'mask_dir': f"{data_root}/implicit_reasoning/masks",
        'gt_dir': f"{data_root}/implicit_reasoning/edited_images",
        'res_dir': f"{model_results_root}/{model_name}_implicit"
    })

    # 3. Multi-Turn / Multi-Object
    multi_tasks = [
        ('multi_turn_remove', ['turn1_remove', 'turn2_remove']),
        ('multi_turn_add', ['turn1_add', 'turn2_add']),
        ('multi_object_remove', ['multi_object_remove']),
        ('multi_object_add', ['multi_object_add'])
    ]

    for main_task, sub_dirs in multi_tasks:
        for sub_dir in sub_dirs:
            # 强制使用根目录下的 main_task.jsonl (例如 multi_turn_add.jsonl)
            info_file = f"{data_root}/multi_turn_editing/{main_task}.jsonl"

            # 默认路径 (Original)
            final_res_dir = f"{model_results_root}/{model_name}_multi/{sub_dir}"

            # 如果是 multi_turn_add，优先查找 multi_turn_add_new 结构
            if main_task == 'multi_turn_add':
                priority_path = f"{model_results_root}/{model_name}_multi/multi_turn_add_new/{sub_dir}"
                if os.path.exists(priority_path):
                    final_res_dir = priority_path

            tasks.append({
                'group': 'Multi-object Editing' if 'object' in main_task else 'Multi-turn',
                'subtask': sub_dir,
                'type': 'multi',
                'info_file': info_file,
                'ori_dir': f"{data_root}/multi_turn_editing/{sub_dir}/ori_images",
                'mask_dir': f"{data_root}/multi_turn_editing/{sub_dir}/masks",
                'gt_dir': f"{data_root}/multi_turn_editing/{sub_dir}/edited_images",
                'res_dir': final_res_dir 
            })
    
    return tasks

def main():
    parser = argparse.ArgumentParser(description="Resume-capable CompBench Evaluation")
    parser.add_argument("--model_names", nargs='+', required=True, help="Models to evaluate")
    parser.add_argument("--tasks", nargs='+', default=['all'], help="Filter tasks")
    parser.add_argument("--metric", default="all", choices=["all", "LC-T", "LC-I", "PSNR", "SSIM", "LPIPS"], 
                        help="Specify a single metric to evaluate, or 'all' for everything.")
    parser.add_argument("--data_root", default="/data/jbh/CompBench/tasks", help="GT Data Root")
    parser.add_argument("--results_root", default="/data/jbh/CompBench/editing_results", help="Model Inference Output Root")
    parser.add_argument("--output_dir", default="evaluation_outputs", help="Directory to save CSV results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="Enable resume mode. Saves detailed per-image CSVs.")
    args = parser.parse_args()

    evaluator = CompBenchEvaluator(device=args.device, metric=args.metric)
    
    # 根据指定的 metric 设置列名和文件后缀名
    metric_suffix = f"_{args.metric}" if args.metric != 'all' else ""
    target_metrics = [args.metric] if args.metric != 'all' else ['LC-T', 'LC-I', 'PSNR', 'SSIM', 'LPIPS']
    
    for model_name in args.model_names:
        print(f"\n{'='*40}\nEvaluating Model: {model_name} | Metric: {args.metric}\n{'='*40}")
        
        model_save_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)
        
        model_subtask_rows = []
        model_res_root = os.path.join(args.results_root, model_name)
        all_configs = get_all_task_configs(args.data_root, model_res_root, model_name)

        target_configs = []
        if 'all' in args.tasks:
            target_configs = all_configs
        else:
            for cfg in all_configs:
                is_match = False
                for t_arg in args.tasks:
                    if (t_arg.lower() in cfg['subtask'].lower()) or \
                       (t_arg.lower() in cfg['group'].lower()) or \
                       (t_arg.lower() == cfg['type']):
                        is_match = True
                        break
                if is_match:
                    target_configs.append(cfg)

        if not target_configs:
            continue

        for task_cfg in target_configs:
            subtask_name = task_cfg['subtask']
            # 将 metric 名字带入细分文件，避免测单指标覆盖测全指标的数据
            task_detailed_csv_path = os.path.join(model_save_dir, f"{subtask_name}{metric_suffix}_detailed.csv")
            
            if not os.path.exists(task_cfg['info_file']):
                print(f"Skipping {subtask_name}: Info file missing at {task_cfg['info_file']}.")
                continue

            # 读取 Metadata
            try:
                if task_cfg['info_file'].endswith('.json'):
                    with open(task_cfg['info_file'], 'r') as f:
                        items = json.load(f)
                else:
                    with open(task_cfg['info_file'], 'r') as f:
                        items = [json.loads(line) for line in f]
            except Exception as e:
                print(f"Error reading info file {task_cfg['info_file']}: {e}")
                continue

            processed_imgs = set()
            if args.resume:
                if os.path.exists(task_detailed_csv_path):
                    try:
                        df_exist = pd.read_csv(task_detailed_csv_path)
                        if 'Image_Name' in df_exist.columns:
                            processed_imgs = set(df_exist['Image_Name'].astype(str))
                        print(f"[{subtask_name}] Resuming... Found {len(processed_imgs)} processed images.")
                    except:
                        pass
            
            items_to_process = []
            for item in items:
                img_name = item.get('image_path') or item.get('image_name')
                if img_name and str(img_name) not in processed_imgs:
                    items_to_process.append(item)
            
            # --- 开始处理 ---
            if len(items_to_process) == 0 and len(items) > 0:
                print(f"[{subtask_name}] All images processed.")
            elif len(items) == 0:
                print(f"[{subtask_name}] No items in metadata.")
                continue
            else:
                memory_metrics = defaultdict(list)
                file_handle = None
                writer = None
                
                if args.resume:
                    file_exists = os.path.exists(task_detailed_csv_path)
                    file_handle = open(task_detailed_csv_path, 'a', newline='', encoding='utf-8')
                    fieldnames = ['Image_Name'] + target_metrics
                    writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()

                for item in tqdm(items_to_process, desc=subtask_name, leave=False):
                    img_name = item.get('image_path') or item.get('image_name')
                    
                    # -----------------------------------------------------------------
                    # 按照新规则严格提取 Caption
                    # -----------------------------------------------------------------
                    captions_to_eval = []
                    
                    if task_cfg['type'] == 'local' or task_cfg['type'] == 'implicit':
                        cap = item.get('caption')
                        if cap: captions_to_eval.append(cap)
                            
                    elif task_cfg['type'] == 'multi':
                        if 'turn1' in subtask_name:
                            cap = item.get('turn1_caption')
                            if cap: captions_to_eval.append(cap)
                        elif 'turn2' in subtask_name:
                            cap = item.get('turn2_caption')
                            if cap: captions_to_eval.append(cap)
                        elif 'multi_object' in subtask_name:
                            cap1 = item.get('caption1')
                            cap2 = item.get('caption2')
                            if cap1: captions_to_eval.append(cap1)
                            if cap2: captions_to_eval.append(cap2)
                            
                    if not captions_to_eval: 
                        captions_to_eval = [" "] # 兜底

                    ori_p = os.path.join(task_cfg['ori_dir'], img_name)
                    edited_p = os.path.join(task_cfg['res_dir'], img_name)
                    gt_p = os.path.join(task_cfg['gt_dir'], img_name)
                    mask_p = os.path.join(task_cfg['mask_dir'], img_name)
                    if 'multi_object' in subtask_name and not os.path.exists(mask_p):
                        base_name = os.path.splitext(img_name)[0]
                        mask_p = os.path.join(task_cfg['mask_dir'], f"{base_name}_t1.png")

                    ori, edited, mask, gt = evaluator.preprocess_images(ori_p, edited_p, mask_p, gt_p)

                    # 计算指标（传入的是列表形式的 captions_to_eval）
                    res = evaluator.calculate_metrics(ori, edited, mask, gt, captions_to_eval)
                    
                    if res:
                        res_rounded = {k: round(v, 3) if not np.isnan(v) else np.nan for k, v in res.items()}
                        if args.resume:
                            row = {'Image_Name': img_name}
                            row.update(res_rounded)
                            writer.writerow(row)
                        else:
                            for k, v in res.items():
                                if not np.isnan(v):
                                    memory_metrics[k].append(v)

                if file_handle:
                    file_handle.close()

            # --- 计算本任务的 Mean ---
            avg_row = None
            if args.resume:
                if os.path.exists(task_detailed_csv_path):
                    try:
                        df_task = pd.read_csv(task_detailed_csv_path)
                        if not df_task.empty:
                            avg_row = {
                                'Model': model_name,
                                'Group': task_cfg['group'],
                                'Subtask': subtask_name,
                                'Count': len(df_task)
                            }
                            for metric_name in target_metrics:
                                avg_row[metric_name] = round(df_task[metric_name].mean(), 3)
                    except Exception as e:
                        print(f"Error calculating mean from csv: {e}")
            else:
                if memory_metrics:
                    avg_row = {
                        'Model': model_name,
                        'Group': task_cfg['group'],
                        'Subtask': subtask_name,
                        'Count': len(list(memory_metrics.values())[0]) if memory_metrics else 0
                    }
                    for metric_name in target_metrics:
                        if metric_name in memory_metrics and memory_metrics[metric_name]:
                            avg_row[metric_name] = round(np.mean(memory_metrics[metric_name]), 3)
                        else:
                            avg_row[metric_name] = 0.0

            if avg_row:
                model_subtask_rows.append(avg_row)
                disp_metric = args.metric if args.metric != 'all' else 'LC-T'
                print(f"  -> {subtask_name} Finished. {disp_metric}: {avg_row.get(disp_metric, 0.0):.3f}")

        # --- 所有子任务跑完后，进行聚合计算 ---
        if model_subtask_rows:
            df_current_detailed = pd.DataFrame(model_subtask_rows)
            cols = ['Model', 'Group', 'Subtask', 'Count'] + target_metrics
            df_current_detailed = df_current_detailed[cols]
            
            detailed_path = os.path.join(model_save_dir, f"summary{metric_suffix}_detailed.csv")
            
            if os.path.exists(detailed_path):
                try:
                    df_existing = pd.read_csv(detailed_path)
                    current_subtasks = df_current_detailed['Subtask'].tolist()
                    df_existing = df_existing[~df_existing['Subtask'].isin(current_subtasks)]
                    df_detailed = pd.concat([df_existing, df_current_detailed], ignore_index=True)
                except Exception as e:
                    print(f"Error reading existing detailed summary, creating new one: {e}")
                    df_detailed = df_current_detailed
            else:
                df_detailed = df_current_detailed

            df_detailed.to_csv(detailed_path, index=False)
            print(f"\nUpdated detailed summary at: {detailed_path}")

            aggregated_rows = []
            def calc_macro_mean(df_subset, name):
                if df_subset.empty: return None
                row = {
                    'Model': model_name,
                    'Category': name,
                    'Subtasks_Count': len(df_subset)
                }
                for metric_name in target_metrics:
                    if metric_name in df_subset.columns:
                        row[metric_name] = round(df_subset[metric_name].mean(), 3)
                return row

            df_local = df_detailed[df_detailed['Group'] == 'Local Editing']
            res_local = calc_macro_mean(df_local, "Local Editing")
            if res_local: aggregated_rows.append(res_local)

            df_implicit = df_detailed[df_detailed['Group'] == 'Implicit Reasoning']
            res_implicit = calc_macro_mean(df_implicit, "Implicit Reasoning")
            if res_implicit: aggregated_rows.append(res_implicit)

            df_multi_obj = df_detailed[df_detailed['Group'] == 'Multi-object Editing']
            res_multi_obj = calc_macro_mean(df_multi_obj, "Multi-object Editing")
            if res_multi_obj: aggregated_rows.append(res_multi_obj)

            df_multi_turn = df_detailed[df_detailed['Group'] == 'Multi-turn']
            if not df_multi_turn.empty:
                df_t1 = df_multi_turn[df_multi_turn['Subtask'].str.contains('turn1')]
                res_t1 = calc_macro_mean(df_t1, "Multi-turn (Turn 1)")
                if res_t1: aggregated_rows.append(res_t1)
                
                df_t2 = df_multi_turn[df_multi_turn['Subtask'].str.contains('turn2')]
                res_t2 = calc_macro_mean(df_t2, "Multi-turn (Turn 2)")
                if res_t2: aggregated_rows.append(res_t2)

            if aggregated_rows:
                df_agg = pd.DataFrame(aggregated_rows)
                agg_path = os.path.join(model_save_dir, f"summary{metric_suffix}_averaged.csv")
                df_agg.to_csv(agg_path, index=False)
                print(f"Updated averaged summary at: {agg_path}")
                print("-" * 40)
                print(df_agg.to_string(index=False))
                print("-" * 40)

if __name__ == "__main__":
    main()