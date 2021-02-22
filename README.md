# Graph Barlow Twins
This repository provides the official implementation of the Graph Barlow Twins method.
It also includes the whole experimental pipeline which is built using DVC.

![](assets/model.png)

### How to use?
- create and activate virtual environment (`venv`)
- install dependencies (`pip install -r requirements.txt`)
- *pull all files from the DVC remote (`dvc pull`)

*The whole pipeline should be reproducible without any external data dependencies.
If you want to use precomputed stage outputs, please perform the `dvc pull` command
and it will download all stage artifacts into the `data/ssl/` directory. You don't
need any credentials as a public DVC remote endpoint is used in the DVC configuration
file. The total size of all artifacts is about 170GB.

If you want to use Docker instead of virtual environments, this repo contains also
a ready-to-use Dockerfile:
```bash
docker build -t graph_barlow_twins:latest -f docker/Dockerfile .

# This script assumes that you will use a GPU for computation acceleration.
./docker/run-docker-gpu.sh "<gpu-id>"

# If you do not have access to a GPU, use the following script:
./docker/run-docker-cpu.sh
```

## Training & evaluation
We implement all our models using the PyTorch-Geometric library and use DVC
(Data Version Control) for model versioning. DVC enables to run all experiments
in a single command and ensure better reproducibility. To reproduce the whole
pipeline run: `dvc repro` and to execute a single stage use: `dvc repro -f -s <stage name>`

There are following stages (see `dvc.yaml` file):
- `preprocess_dataset@<dataset_name>` – downloads the <dataset_name> dataset; if applicable, generates the node splits for train/val/test,
- `full_batch_hps@<dataset_name>` – runs the augmentation hyperparameter search  for a given dataset in the full-batch case,
- `full_batch_train@<dataset_name>`, `batched_train@<dataset_name>` – trains and evaluates the G-BT model for a given dataset in the full-batch case and the batched scenario, respectively,
- `batched_hps_ogbn_products` – runs the augmentation hyperparameter search for the ogb-products dataset in the batched scenario,
- `batched_train_ogbn_products` – trains and evaluated the G-BT model for the ogb-products dataset in the batched scenario,
- `compare_augmentation_hyperparameter_sets` – loads all full-batch augmentation hyperparameter results, compares the case when using the same or different sets of hyperparameters to generate both graph views,
- `compare_running_times` – computes the average running time of a training epoch for the following methods: DeepWalk, DGI, MVGRL, GRACE, BGRL and G-BT,
- `train_bgrl_full_batch@<dataset_name>` – trains and evaluates the BGRL model in the full-batch case for WikiCS, Amazon-CS, Amazon-Photo, and Coauthor-CS,
- `bgrl_hps_batched@ogbn-products` – runs the augmentation hyperparameter search for BGRL using the ogb-products dataset,
- `bgrl_batched_train@ogbn-products` – trains and evaluates the BGRL model for the ogb-products dataset,
- `evaluate_features_products` – evaluates the performance of ogb-products’ raw node features,
- `evaluate_deepwalk_products` – evaluates the performance of DeepWalk on the ogb-products dataset; additionally the case of DeepWalk features concatenated with raw node features is also evaluated.


All hyperparameters described in Appendix A are stored in configuration files in
the `experiments/configs/` directory, whereas the experimental Python scripts are
placed in the `experiments/scripts/` directory.

## Reference
If you make use Graph Barlow Twins in your research, please cite it using the following entry:

```
@misc{bielak2021graph,
      title={Graph Barlow Twins: A self-supervised representation learning framework for graphs},
      author={Piotr Bielak and Tomasz Kajdanowicz and Nitesh V. Chawla},
      year={2021},
      eprint={2106.02466},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License
MIT
