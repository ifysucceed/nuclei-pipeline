# Biomedical Image Analysis Pipeline -- Nuclei Segmentation

Code submission for the "Data Analysis with AI" module assignment. Modality: fluorescence
microscopy (DAPI-style stained nuclei), synthetic `nuclei_dataset` (80 train / 20 val /
12 test / 4 corrupted), originally from
[Nickolay-K/Assingnment-3-dataset](https://github.com/Nickolay-K/Assingnment-3-dataset).

Pipeline: **raw image -> segmentation -> quantitative region features -> structured JSON
record -> short narrative**, combining a local multimodal LLM (Task 1), classical image
processing + a numbers-only LLM interpretation (Task 2), a trained U-Net (Task 3), and a
hybrid pipeline run on unseen test images (Task 4), plus three extensions: a loss-function
ablation, a corruption-robustness trace, and a foundation-model (MedSAM) comparison.

## Repository structure

```
.
├── pipeline_utils.py               # All reusable pipeline functions (documented)
├── nuclei_pipeline.ipynb           # Main notebook: Tasks 1-4 + extensions, run end-to-end
├── requirements.txt                # Python dependencies
├── .gitignore
├── LICENSE
├── README.md                       # This file
├── nuclei_dataset/                 # Dataset (unmodified, as downloaded from the source repo)
│   ├── train/  val/  test/  test_corrupted/
│   ├── metadata.csv, dataset_summary.json, make_dataset.py, README.md
├── outputs/                        # Saved results the notebook produces (committed, real)
│   ├── figures/                    # PNG copy of every plot the notebook produces, for the report
│   │   ├── task1_preprocessing_example.png, task1_sample_grid.png, task1_intensity_histogram.png
│   │   ├── task2_classical_pipeline.png
│   │   ├── task3_loss_dice_curves.png, task3_val_panels.png
│   │   └── extension_loss_ablation.png, extension_robustness_trace.png, extension_medsam_comparison.png
│   ├── unet_bcedice.pt                       # Trained U-Net weights
│   ├── history_bcedice.json                  # Training/validation curves
│   ├── val_dices_bcedice.npy, val_ious_bcedice.npy
│   ├── loss_ablation_results.json, loss_ablation_summary_table.csv
│   ├── task1_vlm_runs.json                   # Real naive + engineered VLM prompt outputs
│   ├── task2_numbers_first_response.json     # Real numbers-first LLM output
│   ├── task4_hybrid_pipeline_records.csv     # Aggregated Task 4 JSON records + narratives
│   ├── otsu_vs_gt_train.csv                  # Otsu vs ground truth, all 80 training images
│   ├── task4_unet_vs_otsu_vs_gt_test.csv     # Otsu vs U-Net vs ground truth, all 12 test images
│   ├── gt_mask_undercounting_evidence.csv    # Connected components on the ground-truth mask itself
│   ├── medsam_vs_unet_comparison.csv         # Extension C results
│   └── extension_robustness_trace.csv        # Extension B results
├── report/
│   └── report.pdf                  # 4-page write-up (methods, figures, discussion, Q&A)
└── MedSAM/                         # NOT committed (see .gitignore) -- third-party library
                                     # cloned locally to run Extension C; see setup below
```

`outputs/` is committed with real results already in it, so
the marker can see genuine output without re-running anything and even when rer-run
outputs remains the same. `outputs/figures/` holds a
standalone PNG of every plot the notebook produces, each plot cell both displays inline
*and* saves a copy there, so figures are available as files for the report without having
to screenshot the notebook. Re-running the notebook overwrites these files with a fresh
run of the same code.

## Setup

```bash
pip install -r requirements.txt
```

### Installing Ollama and pulling the required models

Tasks 1, 2 and 4 call a local LLM through [Ollama](https://ollama.com), using the same
`ollama` Python client (`from ollama import chat`) taught in the module's Lab 2. To enable
these steps:

1. **Install Ollama** (skip if already installed, e.g. from Lab 2):
   - Windows / macOS: download the installer from [ollama.com/download](https://ollama.com/download).
   - Linux (or inside Colab, as in Lab 2): `curl -fsSL https://ollama.com/install.sh | sh`
2. **Start the server** (the desktop app does this automatically; from a terminal you can
   also run `ollama serve`).
3. **Pull the two models this assignment needs:**
   ```bash
   ollama pull llama3.2-vision   # ~7.9 GB, used in Task 1 (image description)
   ollama pull llama3.2          # used in Task 2 and Task 4 (text-only interpretation)
   ```
4. Re-run the notebook's Setup cell -- it prints `Ollama vision model ready: True` /
   `Ollama text model ready: True` once both are pulled.

**Known issue (resolved for this project):** some recent Ollama versions (v0.30.0
onward) cannot run `llama3.2-vision` at all -- it downloads fine but crashes with
`unknown model architecture: 'mllama'` the moment it's used. This is a confirmed upstream
regression (Ollama's own v0.30.0 release notes list `llama3.2-vision is not yet supported`
under Known Issues; see also github.com/ollama/ollama/issues/16490 and /16547). If you hit
this, two fixes:
- Install [Ollama v0.24.0](https://github.com/ollama/ollama/releases/tag/v0.24.0) (the
  last version before this regression, there is no v0.25-v0.29), or
- Set `VISION_MODEL = "llava:7b"` in the notebook's Setup cell and
  `ollama pull llava:7b` instead (explicitly sanctioned as a substitute in Lab 2).

If Ollama isn't available at all, the notebook still runs end-to-end: the LLM cells detect
this and print a skip message instead of failing, so Tasks 1/2/4's non-LLM steps (EDA,
classical features, U-Net masks) still execute and produce real output.

### Enabling the foundation-model extension (MedSAM)

Extension C compares the U-Net against MedSAM. This needs its own local setup (separate
from Ollama):
```
git clone https://github.com/bowang-lab/MedSAM
cd MedSAM && pip install -e .
```
then download `medsam_vit_b.pth` (~357MB) from the checkpoint's Google Drive link
(https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN) and place it at
`MedSAM/work_dir/MedSAM/medsam_vit_b.pth`. Like the Ollama cells, this extension detects
whether it's set up and skips gracefully with setup instructions if not, rather than
crashing the rest of the notebook.

**Why `MedSAM/` is not committed to this repository:** cloning it creates a second, nested
Git repository (it has its own `.git` folder), which Git does not handle cleanly as a
regular subfolder of another repository. Its checkpoint is also ~357MB, over GitHub's
100MB hard per-file limit, so committing it would fail outright regardless. None of the
code inside `MedSAM/` is this project's own work either it is a third-party library
installed to run one extension. What *is* this project's own work, and *is* committed
under `outputs/`, is the actual output of running it: `medsam_vs_unet_comparison.csv` and
`outputs/figures/extension_medsam_comparison.png`.

## Running

Open `nuclei_pipeline.ipynb` in Jupyter/VS Code and run all cells top to bottom
(`Kernel > Restart & Run All`). Total runtime on CPU is roughly 25-30 minutes, dominated by
the U-Net training (Task 3, approximately 9 - 11 min for 20 epochs) and the loss-ablation extension
(approximately 13 min for 3x10 epochs); the MedSAM extension adds more time on top of this if enabled
(pretrained ViT-B image encoder inference per candidate box, on CPU). The notebook creates
`outputs/` automatically if it doesn't already exist (Git does not track empty folders, so
a fresh clone needs this).

## Notes for the marker

- All numeric results in `report/report.pdf` (U-Net Dice/IoU, Otsu-vs-ground-truth
  comparison, loss ablation table, robustness trace) were produced by an actual run of
  this exact code against the real dataset.
- **U-Net architecture:** `pipeline_utils.py`'s `UNet` class is the architecture
  **provided in the module's Lab 4** (`Lab4_CNN_unet_segmentation_SOLUTIONS.ipynb`, "The
  U-Net Architecture" section), reproduced unchanged (class/variable names included) so
  it is traceable back to the source: 3 encoder stages, a bottleneck, 3 decoder stages
  with skip connections, approximately 483K parameters. It is trained here on this assignment's real
  dataset rather than Lab 4's own synthetic ellipses, using the same loss (BCE+Dice),
  optimiser (Adam, lr=1e-3), and batch size (8) as the lab.

- **`density_class` covers all four of the dataset's regimes:** the README defines
  sparse/normal/dense/**clustered** ("touching nuclei"), not three. Since "clustered" is a
  shape property whose object-count range overlaps normal and dense, count alone cannot
  detect it; `summarize_region_table()` also checks the area coefficient of variation
  (touching nuclei merge into components of very unequal size). Validated against all 80
  training images' ground-truth density labels: 72.5% overall 4-class accuracy, 69%
  (9/13) recall specifically on clustered images, a real but imperfect classical signal,
  documented as such rather than overclaimed.

- **Extension C (foundation model)** compares MedSAM against the trained U-Net; see
  "Enabling the foundation-model extension" above for setup. Since MedSAM only accepts
  one bounding box per call (segments one object per box) while our task has dozens of
  nuclei per image, Otsu's connected components are used purely as candidate box
  *proposals*, and the union of MedSAM's per-box masks is taken as its answer per image.

- **Grayscale conversion:** `pipeline_utils.py`'s `to_grayscale_resized()` extracts the
  blue channel directly rather than using a generic RGB->grayscale luminosity formula.
  This is deliberate, not an oversight: `make_dataset.py` (provided with the assignment
  data) shows this dataset's signal is written almost entirely into the blue channel
  (DAPI-style staining), and a generic luminosity formula gives blue only ~11% weight
  (tuned for natural photographs, not fluorescence data). This choice is validated
  directly against `metadata.csv`'s ground-truth `mean_intensity`/`area_fraction`
  columns in Task 1/2 of the notebook (mean absolute error 0.0005 and 0.0023 across all
  80 training images) -- see the docstring in `pipeline_utils.py` for details.
- The three LLM steps (Task 1 direct description, Task 2 numbers-first interpretation,
  Task 4 hybrid narrative) require Ollama running locally with the two models pulled (see
  above). The exact prompts used at every LLM step are reproduced verbatim in both the
  notebook and the report, as required by the assignment brief.
