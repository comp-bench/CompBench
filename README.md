<div align="center">

# CompBench: Benchmarking Complex Instruction-guided Image Editing

[![Paper](https://img.shields.io/badge/arXiv-2505.12200-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.12200)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-4b44ce?logo=opencv&logoColor=white)](https://cvpr.thecvf.com/)
[![Project Page](https://img.shields.io/badge/Project-Page-green?logo=github&logoColor=white)](https://comp-bench.github.io/)
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/BohanJia/CompBench)

Bohan Jia<sup>1\*</sup>, Wenxuan Huang<sup>1,2\*</sup>, Yuntian Tang<sup>1\*</sup>, Junbo Qiao<sup>1</sup>, Jincheng Liao<sup>1</sup>, Shaosheng Cao<sup>3†</sup>, Fei Zhao<sup>3</sup>, Zhaopeng Feng<sup>4</sup>, Zhouhong Gu<sup>5</sup>, Zhenfei Yin<sup>6</sup>, Lei Bai<sup>7</sup>, Wanli Ouyang<sup>2</sup>, Lin Chen<sup>8</sup>, Fei Zhao<sup>9</sup>, Zihan Wang<sup>1</sup>, Yuan Xie<sup>1</sup>, Shaohui Lin<sup>1†</sup>

<sup>1</sup>East China Normal University &nbsp; <sup>2</sup>The Chinese University of Hong Kong &nbsp; <sup>3</sup>Xiaohongshu Inc. &nbsp; <sup>4</sup>Zhejiang University &nbsp; <sup>5</sup>Fudan University &nbsp; <sup>6</sup>University of Oxford &nbsp; <sup>7</sup>Shanghai Jiao Tong University &nbsp; <sup>8</sup>University of Science and Technology of China &nbsp; <sup>9</sup>Nanjing University

*\* Equal contribution &nbsp;&nbsp; † Corresponding author*

**CVPR 2026**

</div>

---

<div align="center">
  <img src="https://comp-bench.github.io/static/images/tasks-1.png" alt="CompBench Task Overview" width="90%">
  <p><em>CompBench covers 9 editing tasks spanning 5 major categories, featuring fine-grained, complex instructions that challenge state-of-the-art image editing models.</em></p>
</div>

---

## News

- **[2026-04]** CompBench is accepted to **CVPR 2026** as a poster presentation.
- **[2025-05]** Paper released on [arXiv](https://arxiv.org/abs/2505.12200).
- **[2025-05]** Dataset released on [HuggingFace](https://huggingface.co/datasets/BohanJia/CompBench).
- **[2025-05]** Project page live at [comp-bench.github.io](https://comp-bench.github.io/).

---

## Abstract

While real-world applications increasingly demand intricate scene manipulation, existing instruction-guided image editing benchmarks often oversimplify task complexity and lack comprehensive, fine-grained instructions. To bridge this gap, we introduce **CompBench**, a large-scale benchmark specifically designed for complex instruction-guided image editing.

CompBench features challenging editing scenarios that incorporate **fine-grained instruction following**, **spatial and contextual reasoning**, thereby enabling comprehensive evaluation of image editing models' precise manipulation capabilities. We propose an **MLLM-human collaborative framework** with tailored task pipelines, and an **instruction decoupling strategy** that disentangles editing intents into four key dimensions: *location*, *appearance*, *dynamics*, and *objects*.

Extensive evaluations reveal that CompBench exposes fundamental limitations of current image editing models and provides critical insights for next-generation instruction-guided image editing.

<div align="center">
  <img src="https://comp-bench.github.io/static/images/Overall-1.png" alt="CompBench Pipeline" width="85%">
  <p><em>Overview of the MLLM-human collaborative data construction pipeline.</em></p>
</div>

---

## Task Categories

CompBench organizes editing into **5 major categories** covering **9 distinct tasks**:

| Category | Tasks | Description |
|---|---|---|
| **Local Editing** | Object Removal, Addition, Replacement | Precise object-level manipulation within a scene |
| **Multi-editing** | Multi-turn Editing, Multi-object Editing | Chained or simultaneous edits across multiple targets |
| **Action Editing** | Dynamic State Modification | Modifying object actions, poses, and dynamic states |
| **Scene Spatial Editing** | Location Editing, Viewpoint Editing | Spatial repositioning and perspective change |
| **Complex Reasoning** | Implicit Reasoning | Edits requiring contextual or commonsense inference |

<div align="center">
  <img src="https://comp-bench.github.io/static/images/act_loc_view.jpg" alt="Action, Location, and Viewpoint Editing Examples" width="85%">
  <p><em>Examples of action, location, and viewpoint editing tasks.</em></p>
</div>

<div align="center">
  <img src="https://comp-bench.github.io/static/images/local_editing.jpg" alt="Local Editing Examples" width="85%">
  <p><em>Examples of local editing tasks (removal, addition, replacement).</em></p>
</div>

---

## Dataset Statistics

| Statistic | Value |
|---|---|
| Total image-instruction pairs | **3,000+** |
| Editing task categories | **5** |
| Distinct editing tasks | **9** |
| Average objects per image | **13.58** |
| Occlusion rate | **98.47%** |
| Out-of-frame rate | **86.38%** |
| Models evaluated | **15+** |

The high occlusion and out-of-frame rates reflect the real-world complexity of CompBench scenes, making it significantly more challenging than prior benchmarks.

---

## Evaluation Metrics

CompBench uses a suite of complementary metrics to capture both editing quality and background preservation:

| Metric | Name | Measures |
|---|---|---|
| **LC-T** | Local CLIP Text Score | Instruction-following fidelity in the edited region |
| **LC-I** | Local CLIP Image Similarity | Edited region accuracy compared to ground truth |
| **PSNR** | Peak Signal-to-Noise Ratio | Background reconstruction quality |
| **SSIM** | Structural Similarity Index | Structural fidelity of background |
| **LPIPS** | Learned Perceptual Image Patch Similarity | Perceptual background consistency |

---

## Leaderboard

<div align="center">
  <img src="https://comp-bench.github.io/static/images/overall_ranking.png" alt="Overall Model Ranking" width="80%">
  <p><em>Overall model ranking across all CompBench tasks.</em></p>
</div>

<div align="center">
  <img src="https://comp-bench.github.io/static/images/top_models_radar.png" alt="Top Models Radar Chart" width="60%">
  <p><em>Radar chart comparing top-performing models across evaluation dimensions.</em></p>
</div>

<div align="center">
  <img src="https://comp-bench.github.io/static/images/model_ssim_comparison_bubble.png" alt="SSIM Comparison Bubble Chart" width="75%">
  <p><em>SSIM comparison bubble chart across evaluated models.</em></p>
</div>

CompBench evaluates 15+ models spanning the full spectrum of instruction-guided image editing, from early approaches (InstructPix2pix) to the latest generation (FLUX.1 Kontext, Bagel, Qwen-Image-Edit). Results demonstrate that even state-of-the-art models struggle with complex, fine-grained editing instructions, highlighting the benchmark's discriminative power.

<div align="center">
  <img src="https://comp-bench.github.io/static/images/Comparison.jpg" alt="Qualitative Comparison" width="90%">
  <p><em>Qualitative comparison of editing results across models on representative CompBench samples.</em></p>
</div>

---

## Installation

**Requirements:** Python 3.8+, PyTorch with CUDA support recommended.

```bash
# Clone the repository
git clone https://github.com/BhJia/CompBench.git
cd CompBench

# Install dependencies
pip install torch torchvision
pip install torchmetrics[multimodal] transformers pillow tqdm pandas numpy
```

---

## Quick Start

### 1. Download the Dataset

Download the CompBench dataset from HuggingFace to your local filesystem:

```bash
python download_from_hf.py
```

This script fetches the dataset from [`BohanJia/CompBench`](https://huggingface.co/datasets/BohanJia/CompBench) and organizes it into the expected local directory structure under `./tasks/`.

The dataset is organized by task:

```
tasks/
├── add/                  # Object addition
├── remove/               # Object removal
├── replace/              # Object replacement
├── act_loc_view/         # Action, location, viewpoint editing
├── implicit_reasoning/   # Complex reasoning
└── multi_turn_editing/   # Multi-turn editing
```

Each task directory contains source images, editing instructions, segmentation masks, and ground-truth edited images.

### 2. Prepare Model Outputs

Place your model's edited images under `./editing_results/` following this structure:

```
editing_results/
└── <model_name>/
    └── <task_name>/
        └── <image_id>.png
```

### 3. Run Evaluation

```bash
python eval_all.py \
  --model_names model1 model2 \
  --tasks all \
  --metric all \
  --data_root ./tasks \
  --results_root ./editing_results \
  --output_dir ./eval_results \
  --resume
```

See [`eval_all.sh`](eval_all.sh) for a concrete example with multiple models and metrics.

**Arguments:**

| Argument | Description | Default |
|---|---|---|
| `--model_names` | Space-separated list of model names to evaluate | required |
| `--tasks` | Task(s) to evaluate (`all` or specific task names) | `all` |
| `--metric` | Metric(s) to compute (`all`, `LC-T`, `LC-I`, `PSNR`, `SSIM`, `LPIPS`) | `all` |
| `--data_root` | Path to the dataset root directory | `./tasks` |
| `--results_root` | Path to model editing results | `./editing_results` |
| `--output_dir` | Directory to save evaluation outputs | `./eval_results` |
| `--resume` | Skip already-evaluated samples | flag |

---

## Dataset Management

### Download from HuggingFace

```bash
python download_from_hf.py
```

Downloads the full CompBench dataset from HuggingFace Hub to the local `./tasks/` directory.

### Update HuggingFace Dataset

```bash
python update_hf_dataset.py
```

Syncs local metadata changes back to the HuggingFace dataset repository. Useful for maintainers contributing new annotations or corrections.

---

## Repository Structure

```
CompBench/
├── tasks/                    # Dataset organized by editing task
│   ├── add/
│   ├── remove/
│   ├── replace/
│   ├── act_loc_view/
│   ├── implicit_reasoning/
│   └── multi_turn_editing/
├── eval_all.py               # Main evaluation script
├── eval_all.sh               # Example evaluation command
├── download_from_hf.py       # Download dataset from HuggingFace
└── update_hf_dataset.py      # Sync local metadata to HuggingFace
```

---

## Citation

If you find CompBench useful in your research, please consider citing:

```bibtex
@article{jia2025compbench,
  title={Compbench: Benchmarking complex instruction-guided image editing},
  author={Jia, Bohan and Huang, Wenxuan and Tang, Yuntian and Qiao, Junbo and Liao, Jincheng and Cao, Shaosheng and Zhao, Fei and Feng, Zhaopeng and Gu, Zhouhong and Yin, Zhenfei and others},
  journal={arXiv preprint arXiv:2505.12200},
  year={2025}
}
```

---

## Contact

For questions or feedback, please open an issue on [GitHub](https://github.com/BhJia/CompBench/issues) or reach out via the [project page](https://comp-bench.github.io/).

---

<div align="center">
  <sub>CompBench is accepted to CVPR 2026. &nbsp;|&nbsp; <a href="https://arxiv.org/abs/2505.12200">Paper</a> &nbsp;|&nbsp; <a href="https://comp-bench.github.io/">Project Page</a> &nbsp;|&nbsp; <a href="https://huggingface.co/datasets/BohanJia/CompBench">Dataset</a></sub>
</div>
