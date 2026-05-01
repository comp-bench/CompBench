python eval_all.py \
  --model_names anyedit bagel cosxl flux \
  --tasks all \
  --metric all \
  --data_root ./tasks \
  --results_root ./editing_results \
  --output_dir ./eval_results \
  --device cuda \
  --resume