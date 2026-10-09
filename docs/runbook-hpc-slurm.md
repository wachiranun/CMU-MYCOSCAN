# Training MycoScan on CMU HPC ERAWAN with Slurm

From an empty `/project/wachiranun.sir/` to finished runs: clone the repository, build the Python environment, move the images, prefetch pretrained weights, submit GPU jobs, watch them, and copy the results back.

| | |
|---|---|
| Cluster | ERAWAN, `erawan.cmu.ac.th` |
| Scheduler | Slurm |
| Repository | `wachiranun/CMU-MYCOSCAN` |
| Written | 2026-10-08, against commit `5f2a407` |

Contents

0. [ERAWAN at a glance](#0-erawan-at-a-glance)
1. [Before you log in](#1-before-you-log-in)
2. [Log in and lay out /project](#2-log-in-and-lay-out-project)
3. [Clone the repository](#3-clone-the-repository)
4. [Python environment](#4-python-environment)
5. [Move the data](#5-move-the-data)
6. [Prefetch pretrained weights](#6-prefetch-pretrained-weights-on-the-login-node)
7. [Smoke-test job](#7-smoke-test-job)
8. [Training job template](#8-training-job-template)
9. [Plan 1: OpenFungi pilot](#9-plan-1-openfungi-pilot-stage-by-stage)
10. [Plan 2: CMU main study](#10-plan-2-cmu-main-study)
11. [Monitor jobs](#11-monitor-jobs)
12. [Bring results back](#12-bring-results-back)
13. [Troubleshooting](#13-troubleshooting)
14. [Checklist](#14-checklist)

## 0. ERAWAN at a glance

Facts taken from the ERAWAN wiki on 2026-10-08. Verify the live numbers with `sinfo` and `scontrol show partition gpu` once logged in, since limits change.

| | |
|---|---|
| Login | `ssh wachiranun.sir@cmu.ac.th@erawan.cmu.ac.th`. CMU network or CMU VPN only. Password is the CMU account password. |
| Storage | `/home`: 1 TB archive disk, slow. `/project`: 200 GB per user, NVMe parallel file system, fast. Work here. |
| Internet | Login node has internet. **Compute nodes do not.** Every download (pip, timm weights) happens on the login node. |
| Default partition | `gpu` is the default if none is given. Partition names are lower-case in Slurm: `cpu`, `gpu`, `mixed`, `gpu-h100`. |

| Partition | Hardware | Time limit | Max CPU per job | Running jobs per user | Queued jobs per user | Use it for |
|---|---|---|---|---|---|---|
| `gpu` | 2 nodes, 16 × NVIDIA A100 80 GB, 64 CPUs | 168 h | 4 | 2 | 3 | All training and evaluation. One A100 per job is plenty for this code. |
| `gpu-h100` | 1 node, 4 × H100 80 GB, 128 CPUs | 120 h | 64 | 1 | 2 | Sweeps or multi-seed runs that want many data-loader workers. |
| `mixed` | 1 node, 8 × A100, 128 CPUs | 24 h | 64 | 1 | 2 | Short GPU jobs that need more than 4 CPUs. |
| `cpu` | 2 nodes, 192 CPUs | 168 h | 32 | 2 | 3 | `pytest`, results tables, learning curves, paired comparisons. |

> **Two jobs at a time.** The `gpu` partition runs at most 2 of your jobs and queues at most 3. The queue limit counts every job you own that is still in `squeue`, including jobs held by `--dependency`, so a fourth `sbatch` fails with `QOSMaxSubmitJobPerUserLimit`. Chain stages inside one job or with `--dependency` rather than submitting many small jobs, and let a login-node waiter submit anything beyond three. Section 9 shows all three.

Useful ERAWAN-specific commands: `myquota` (disk use), `mycredit` (credit balance in Baht), `htopc1` to `htopc4` and `nvtopc1` to `nvtopc4` (live CPU and GPU use on compute node 1 to 4).

## 1. Before you log in

1. **ERAWAN account and VPN.** You need an active ERAWAN registration and either the CMU network or the CMU VPN. Check `mycredit` after the first login; training costs credit.
2. **Push the branch you want to run.** The cluster clones from GitHub, so the code must be on `origin`. On 2026-10-08 the full pipeline lives on `ticket-13-metric-block`, which is 4 commits ahead of its remote copy, and `origin/main` still holds only the initial scaffold. From your laptop:

   ```bash
   git push origin ticket-13-metric-block
   ```

   Once the branch is merged, replace `ticket-13-metric-block` with `main` everywhere below.
3. **Know what data goes up.** `openfungi/` (about 7.9 GB), `data/` and `runs/` are git-ignored. They never come with the clone and must be copied separately (section 5).
4. **GitHub access from the cluster.** The repository is public, so an HTTPS clone needs no token. For pushing results or configs back from ERAWAN, create a fine-grained personal access token or add an SSH key there; neither is needed to follow this run book.

## 2. Log in and lay out /project

From PowerShell or any terminal on your laptop:

```bash
ssh wachiranun.sir@cmu.ac.th@erawan.cmu.ac.th
```

Type `yes` at the fingerprint prompt the first time. Everything below runs on the ERAWAN login node unless it says otherwise.

Create one project root and keep repository, data caches, job scripts and logs under it. The repository expects `openfungi/` and `data/` at its own root, so those live inside the clone.

```bash
export PROJ=/project/wachiranun.sir/mycoscan
mkdir -p "$PROJ"/{cache/huggingface,cache/torch,jobs,logs}
cd "$PROJ" && pwd && myquota
```

Target layout after sections 3 to 5:

```text
/project/wachiranun.sir/mycoscan/
├── env.sh                 # sourced by every job: conda env, cache paths
├── jobs/                  # *.sbatch scripts from this run book
├── logs/                  # Slurm stdout/stderr, one pair per job
├── cache/huggingface/     # timm / Hugging Face weights, prefetched on the login node
├── cache/torch/           # torchvision weights (densenet121, resnet50)
└── CMU-MYCOSCAN/          # git clone
    ├── configs/  src/  tests/
    ├── openfungi/macro/<class>/  openfungi/micro/<class>/   # copied, not cloned
    ├── data/openfungi/manifest.csv, splits_v1.csv            # built here or copied
    ├── data/cmu/manifest.csv, splits_cmu.csv                 # Plan 2
    └── runs/<run_name>/                                      # outputs
```

## 3. Clone the repository

```bash
cd /project/wachiranun.sir/mycoscan
git clone --branch ticket-13-metric-block https://github.com/wachiranun/CMU-MYCOSCAN.git
cd CMU-MYCOSCAN
git log --oneline -3          # expect 5f2a407 "Bag panels in explain for attention-MIL..." at the top
```

Later updates:

```bash
cd /project/wachiranun.sir/mycoscan/CMU-MYCOSCAN && git pull --ff-only
```

> **Provenance.** Every `metrics.json` records the git commit and whether the tree was dirty. Keep the clone clean: put experiment overrides in `--set` flags or in new config files that you commit, not in uncommitted edits.

## 4. Python environment

ERAWAN's default `python` is 3.6.8 and its system PyTorch is 1.10; MycoScan needs Python 3.11 or later and `torch==2.14.1`. Build an isolated conda environment with the `anaconda3` module. Do this on the login node, which has internet.

### 4.1 Create the conda environment (once)

```bash
module purge
module load anaconda3
conda config --set auto_activate_base false    # ERAWAN recommends this; run once
conda init bash                                 # once, then log out and back in
exit
```

```bash
ssh wachiranun.sir@cmu.ac.th@erawan.cmu.ac.th
module load anaconda3
conda create -y -n mycoscan python=3.11
conda activate mycoscan
python --version                                # Python 3.11.x
```

### 4.2 Install MycoScan with CUDA wheels

The A100 and H100 nodes run a CUDA 12 driver, so use the `cu126` PyTorch index exactly as the README does. The wheels bundle their own CUDA runtime; you do not need `module load cuda`.

```bash
cd /project/wachiranun.sir/mycoscan/CMU-MYCOSCAN
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install -e .
pip install -e ".[lora]"       # optional: peft, only for finetune = "lora"
pip install -e ".[mlflow]"     # optional: only if you set tracking = "mlflow"
mycoscan env                   # on the login node this prints cuda_available: false; that is expected
```

### 4.3 Write env.sh, sourced by every job

Cache directories go on `/project` so compute nodes read weights from the fast file system. `HF_HUB_OFFLINE` is set in the job scripts, not here, so the prefetch in section 6 can still download.

```bash
cat > /project/wachiranun.sir/mycoscan/env.sh <<'EOF'
# Shared environment for MycoScan on ERAWAN. Source from the login node and from every sbatch script.
export PROJ=/project/wachiranun.sir/mycoscan
export REPO="$PROJ/CMU-MYCOSCAN"
export HF_HOME="$PROJ/cache/huggingface"
export TORCH_HOME="$PROJ/cache/torch"
export MPLBACKEND=Agg                  # headless matplotlib for confusion and reliability plots
export PYTHONUNBUFFERED=1              # log lines reach the .out file as they happen

module purge
module load anaconda3
# conda's shell hook; works whether or not ~/.bashrc was initialised
eval "$(conda shell.bash hook)"
conda activate mycoscan
EOF
source /project/wachiranun.sir/mycoscan/env.sh && which mycoscan
```

## 5. Move the data

The clone has no images. Copy `openfungi/` (and, for Plan 2, `data/cmu/`) from your laptop into the clone on `/project`. Thousands of small files transfer slowly over the VPN, so pack them first.

### 5.1 Pack on the laptop (PowerShell)

```powershell
cd C:\Users\USER\OneDrive\Documents\GitHub\CMU-MYCOSCAN
tar -cf openfungi.tar openfungi            # about 7.9 GB; images are already compressed, so no gzip
Get-FileHash openfungi.tar -Algorithm SHA256 | Format-List
```

### 5.2 Upload with scp (or WinSCP)

```powershell
scp openfungi.tar "wachiranun.sir@cmu.ac.th@erawan.cmu.ac.th:/project/wachiranun.sir/mycoscan/"
```

WinSCP works too: host `erawan.cmu.ac.th`, port 22, CMU account, drag the tar into `/project/wachiranun.sir/mycoscan/`.

### 5.3 Unpack on ERAWAN and verify

```bash
cd /project/wachiranun.sir/mycoscan
sha256sum openfungi.tar                     # compare with the laptop hash
tar -xf openfungi.tar -C CMU-MYCOSCAN && rm openfungi.tar
ls CMU-MYCOSCAN/openfungi/macro CMU-MYCOSCAN/openfungi/micro
find CMU-MYCOSCAN/openfungi -type f | wc -l
myquota
```

### 5.4 Manifests and splits

Two choices. Either copy an existing `data/openfungi/manifest.csv`, `splits_v1.csv` and its `.sha256` sidecar from the laptop (image paths in the manifest are relative to the manifest's folder, so they stay valid), or build them on ERAWAN in section 9.1. Build on ERAWAN unless you already froze a partition: the splits file is pinned to the manifest's hash and must never be regenerated once runs depend on it.

> **Plan 2 data.** CMU isolate images and `data/cmu/splits_cmu.csv` are study data. Copy them only into `/project/wachiranun.sir/`, which is private to you, and keep the sealed test split's sidecar with it. `mycoscan seal` is run by someone other than the modeller.

## 6. Prefetch pretrained weights on the login node

Compute nodes cannot reach the internet, so a training job that needs ImageNet, ImageNet-22k or DINO weights it has never seen will fail. Download every backbone you plan to use once, on the login node, into the caches that `env.sh` points at. The snippet builds each model the same way training does, so the resolved timm tag is identical.

```bash
source /project/wachiranun.sir/mycoscan/env.sh
cd "$REPO"
python - <<'EOF'
from mycoscan.models import build_model
wanted = [
    ("densenet121", "imagenet"),              # configs/openfungi_pretrain.toml, cmu_*.toml
    ("convnext_tiny", "imagenet"),            # openfungi_macro_well_supported, openfungi_stage1_micro
    ("convnext_tiny", "imagenet22k"),
    ("convnext_small", "imagenet"),
    ("tf_efficientnetv2_s", "imagenet"),
    ("resnet50", "imagenet"),
    ("vit_small_patch14_dinov2", "dino"),     # also the OpenFungi pseudo-group embedder
    ("vit_small_patch16_dinov3", "dino"),
    ("convnext_small.dinov3_lvd1689m", "dino"),
    ("vit_base_patch16_224", "imagenet"),
]
for arch, weights in wanted:
    build_model(arch, 1, weights, 224)
    print("cached", arch, weights)
EOF
du -sh "$HF_HOME" "$TORCH_HOME"
```

Add a line to `wanted` whenever a new `arch` or `weights` enters a config. A job that hits a missing weight stops early with a Hugging Face "offline mode" error, so you lose only queue time.

## 7. Smoke-test job

Ten minutes on one A100: confirms the environment sees the GPU, the caches are readable offline, and a tiny synthetic run writes the usual outputs. Submit this before anything that costs real time.

```bash
cat > /project/wachiranun.sir/mycoscan/jobs/00_smoke.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=myco-smoke
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/project/wachiranun.sir/mycoscan/logs/%x_%j.out
#SBATCH --error=/project/wachiranun.sir/mycoscan/logs/%x_%j.err
set -euo pipefail
source /project/wachiranun.sir/mycoscan/env.sh
export HF_HUB_OFFLINE=1
cd "$REPO"
nvidia-smi
mycoscan env                              # expect cuda_available: true, device: NVIDIA A100 80GB
mycoscan make-synthetic --out data/synthetic
mycoscan train --config configs/synthetic/openfungi_pretrain.toml --set num_workers=$SLURM_CPUS_PER_TASK
ls runs
EOF
sbatch /project/wachiranun.sir/mycoscan/jobs/00_smoke.sbatch
```

```bash
squeue -u "$USER"                                   # ST = PD pending, R running
tail -f /project/wachiranun.sir/mycoscan/logs/myco-smoke_*.out   # Ctrl-C to stop following
```

Pass criteria: `cuda_available` is true, the run finishes with a `metrics.json` under `runs/`, and `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS` shows `COMPLETED`. Metrics on synthetic shapes mean nothing; only the plumbing is being tested.

## 8. Training job template

One generic script runs any `mycoscan` subcommand. The config and extra flags come from the `sbatch` command line, so the script is never edited per experiment and the job name says what ran.

```bash
cat > /project/wachiranun.sir/mycoscan/jobs/train.sbatch <<'EOF'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4            # gpu partition allows at most 4; feeds num_workers below
#SBATCH --mem=64G
#SBATCH --time=24:00:00              # raise for multi-seed CV and sweeps; partition cap is 168 h
#SBATCH --output=/project/wachiranun.sir/mycoscan/logs/%x_%j.out
#SBATCH --error=/project/wachiranun.sir/mycoscan/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wachiranun.sir@cmu.ac.th
# Usage: sbatch --job-name=<name> train.sbatch <config.toml> [--set key=value ...]
set -euo pipefail
source /project/wachiranun.sir/mycoscan/env.sh
export HF_HUB_OFFLINE=1              # compute nodes are offline; weights come from the prefetched cache
cd "$REPO"
CONFIG="$1"; shift
echo "job $SLURM_JOB_ID on $(hostname), config $CONFIG, commit $(git rev-parse --short HEAD), extra: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mycoscan env
mycoscan train --config "$CONFIG" \
    --set num_workers="$SLURM_CPUS_PER_TASK" \
    --set amp=true \
    "$@"
EOF
```

### 8.1 Submit

```bash
cd /project/wachiranun.sir/mycoscan
sbatch --job-name=of-pretrain jobs/train.sbatch configs/openfungi_pretrain.toml
sbatch --job-name=of-macro-ws --time=48:00:00 jobs/train.sbatch configs/openfungi_macro_well_supported.toml
sbatch --job-name=of-macro-ws-22k jobs/train.sbatch configs/openfungi_macro_well_supported.toml \
    --set weights=imagenet22k --set run_name=openfungi_macro_ws_in22k
```

Flags on the `sbatch` line override the `#SBATCH` lines in the file, so `--time`, `--partition` and `--job-name` can change per submission. Flags after the config path go to `mycoscan train` unchanged.

### 8.2 What the directives mean here

| Directive | Why this value |
|---|---|
| `--gpus=1` | ERAWAN's documented GPU request. The code trains on one device; more GPUs are not used. |
| `--cpus-per-task=4` | The `gpu` partition's per-job maximum. Passed on as `num_workers`, which the config defaults to 0 (main-process loading, slow on JPEG decoding). |
| `--mem=64G` | Nodes have about 2 TB; 64 GB covers image decoding, bootstrap CIs and the attention-MIL padded bags. Raise if `sacct` shows `OUT_OF_MEMORY`. |
| `--time` | A run killed at the limit leaves no resumable state. Overestimate. Slurm uses the value for backfill priority, so stay realistic. |
| `--set amp=true` | Mixed precision on the A100 roughly halves step time. Drop it for the P0 `small_cnn` reproduction if you want bit-for-bit comparability with CPU runs. |
| `--output=.../%x_%j.out` | Job name and ID in the log name, in a folder outside the clone so logs never show up as a dirty tree. |

### 8.3 CPU-only companion

Tests and the post-processing commands do not need a GPU. ERAWAN asks that nothing heavy run on the login node, so send `pytest` to the `cpu` partition. Light commands such as `mycoscan results` and `mycoscan paired` are fine on the login node.

```bash
cat > /project/wachiranun.sir/mycoscan/jobs/cpu.sbatch <<'EOF'
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/project/wachiranun.sir/mycoscan/logs/%x_%j.out
#SBATCH --error=/project/wachiranun.sir/mycoscan/logs/%x_%j.err
# Usage: sbatch --job-name=<name> cpu.sbatch <command ...>
set -euo pipefail
source /project/wachiranun.sir/mycoscan/env.sh
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$REPO"
"$@"
EOF
sbatch --job-name=pytest jobs/cpu.sbatch pytest -q
sbatch --job-name=pytest-net jobs/cpu.sbatch pytest -q -m network    # needs every weight it uses prefetched
```

## 9. Plan 1: OpenFungi pilot, stage by stage

The experiments P0 to P10 from the October 2026 plan, in the order the data dependencies force. Every stage below assumes the image folders from section 5 and the caches from section 6.

### 9.1 Build the manifest and freeze Pool A / Pool B

The manifest builder embeds every image with DINO to form pseudo-groups, so it runs as a GPU job. The partition step is cheap and deterministic, so it runs on the login node right after. The `cpu.sbatch` wrapper from section 8.3 runs any command; flags on the `sbatch` line override its `#SBATCH` partition, so the same wrapper serves GPU one-offs.

```bash
cd /project/wachiranun.sir/mycoscan
jid=$(sbatch --parsable --job-name=of-manifest --partition=gpu --gpus=1 --cpus-per-task=4 --mem=32G --time=03:00:00 \
      jobs/cpu.sbatch mycoscan build-openfungi --root openfungi --out data/openfungi/manifest.csv \
      --config configs/openfungi_manifest.toml)
echo "manifest job $jid"
```

When it completes, look at the printed per-class summary in the log and skim `data/openfungi/contact_sheets/` (copy a few PNGs to the laptop with scp). Then freeze the partition once and never again:

```bash
source /project/wachiranun.sir/mycoscan/env.sh && cd "$REPO"
mycoscan partition --manifest data/openfungi/manifest.csv --out data/openfungi/splits_v1.csv
cat data/openfungi/splits_v1.csv.sha256
cp data/openfungi/splits_v1.csv data/openfungi/splits_v1.csv.sha256 "$PROJ"/     # a second copy outside the clone
```

### 9.2 P0: leaky versus grouped, then the stage-1 pretrain

Three jobs. With the 2-running, 3-queued limit of the `gpu` partition this fits in one submission round.

```bash
cd /project/wachiranun.sir/mycoscan
sbatch --job-name=p0-leaky   --time=06:00:00 jobs/train.sbatch configs/p0_small_cnn_micro_leaky.toml
sbatch --job-name=p0-grouped --time=06:00:00 jobs/train.sbatch configs/p0_small_cnn_micro_grouped.toml
sbatch --job-name=of-pretrain --time=12:00:00 jobs/train.sbatch configs/openfungi_pretrain.toml
```

### 9.3 Backbone and recipe comparisons (P1 to P5) with dependency chains

Use `--dependency=afterok` so a later job starts only when an earlier one succeeded. `--parsable` makes `sbatch` print just the job ID. Only three jobs can sit in the queue at once, and a job held by `--dependency` counts, so submit the first three directly and hand the fourth to a waiter that runs on the login node until a slot frees.

Write the waiter once. It is a stand-in for `--dependency=afterok:<jobid>` that lives outside the queue: it sleeps until `<jobid>` has left `squeue` and fewer than three of your jobs remain, then submits only if `<jobid>` ended in state `COMPLETED`.

```bash
cat > /project/wachiranun.sir/mycoscan/jobs/submit-after.sh <<'EOF'
#!/bin/bash
# Usage: nohup jobs/submit-after.sh <jobid> <sbatch args ...> > logs/<name>.submit.log 2>&1 &
# Login-node stand-in for --dependency=afterok:<jobid> when the 3-queued-jobs limit leaves
# no room to hold the dependent job in the queue. Waits until <jobid> has left squeue and
# fewer than QUEUE_LIMIT (default 3) of your jobs remain, then submits only if <jobid> COMPLETED.
set -euo pipefail
dep=$1; shift
limit=${QUEUE_LIMIT:-3}
while squeue -h -j "$dep" 2>/dev/null | grep -q . || [ "$(squeue -h -u "$USER" | wc -l)" -ge "$limit" ]; do
  sleep 120
done
state=$(sacct -n -X -j "$dep" -o State | tr -d ' ')
if [ "$state" != "COMPLETED" ]; then
  echo "job $dep ended with state $state; not submitting: sbatch $*" >&2
  exit 1
fi
echo "job $dep COMPLETED; submitting: sbatch $*"
exec sbatch "$@"
EOF
chmod +x /project/wachiranun.sir/mycoscan/jobs/submit-after.sh
```

Then submit the chain. Three go straight to Slurm; the DINOv2 probe waits on the login node for `j2`.

```bash
cd /project/wachiranun.sir/mycoscan
CFG=configs/openfungi_macro_well_supported.toml
j1=$(sbatch --parsable --job-name=ws-convnext-in1k --time=48:00:00 jobs/train.sbatch $CFG)
j2=$(sbatch --parsable --job-name=ws-convnext-in22k --time=48:00:00 jobs/train.sbatch $CFG \
       --set weights=imagenet22k --set run_name=openfungi_macro_ws_convnext_in22k)
j3=$(sbatch --parsable --dependency=afterok:$j1 --job-name=ws-effnet --time=48:00:00 jobs/train.sbatch $CFG \
       --set arch=tf_efficientnetv2_s --set run_name=openfungi_macro_ws_effnetv2s)
nohup jobs/submit-after.sh $j2 --job-name=ws-dinov2-probe --time=12:00:00 jobs/train.sbatch $CFG \
       --set arch=vit_small_patch14_dinov2 --set weights=dino --set finetune=linear_probe \
       --set seeds=[0] --set run_name=openfungi_macro_ws_dinov2_probe > logs/ws-dinov2-probe.submit.log 2>&1 &
echo "$j1 $j2 $j3"; squeue -u "$USER" -o "%.9i %.14j %.3t %.10M %.20R"
```

`nohup` keeps the waiter alive after you log out. `jobs -l` lists it while you are logged in, `pgrep -af submit-after` finds it afterwards, and the submit log shows the job ID once it fires. If the login node was rebooted, check the log before starting the waiter again; a second waiter would submit the probe twice.

> **Why afterok and not afterany.** A failed run should hold back the runs that will later be compared against it; `afterany` would start them regardless and burn credit. A job whose dependency failed stays pending with reason `DependencyNeverSatisfied`; cancel it with `scancel`. The waiter applies the same rule: it logs the failed state and submits nothing.

### 9.4 P7 learning curve as one job

`mycoscan sweep` runs its cells one after another in a single process, so the whole sweep is one Slurm job. Give it the sum of the cells' times. Four cells of 3-seed 5-fold ConvNeXt-Tiny fit comfortably in 72 hours on an A100.

```bash
cat > /project/wachiranun.sir/mycoscan/jobs/sweep.sbatch <<'EOF'
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/project/wachiranun.sir/mycoscan/logs/%x_%j.out
#SBATCH --error=/project/wachiranun.sir/mycoscan/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wachiranun.sir@cmu.ac.th
# Usage: sbatch --job-name=<name> sweep.sbatch configs/sweeps/<sweep>.toml
set -euo pipefail
source /project/wachiranun.sir/mycoscan/env.sh
export HF_HUB_OFFLINE=1
cd "$REPO"
SWEEP="$1"
NAME=$(python -c "import tomllib,sys; print(tomllib.load(open(sys.argv[1],'rb'))['name'])" "$SWEEP")
mycoscan sweep "$SWEEP" || echo "sweep exited 1: at least one cell failed, see runs/$NAME/sweep_summary.json"
mycoscan results "runs/$NAME" --out "runs/$NAME/table.csv"
mycoscan learning-curve "runs/$NAME" --x images --out "runs/$NAME/curve.png"
EOF
sbatch --job-name=p7-curve jobs/sweep.sbatch configs/sweeps/p7_learning_curve.toml
```

The sweep inherits `num_workers=0` from its base config, since cell overrides cannot be injected from the command line. If loading is the bottleneck, add `num_workers = 4` and `amp = true` to the base config file and commit it.

### 9.5 P10: export the Stage-1 checkpoints

Two GPU one-offs through the `cpu.sbatch` wrapper with the partition overridden on the command line:

```bash
cd /project/wachiranun.sir/mycoscan
G="--partition=gpu --gpus=1 --cpus-per-task=4 --mem=64G --time=12:00:00"
sbatch $G --job-name=stage1-micro jobs/cpu.sbatch mycoscan export-stage1 --config configs/openfungi_stage1_micro.toml --set num_workers=4 --set amp=true
sbatch $G --job-name=stage1-macro jobs/cpu.sbatch mycoscan export-stage1 --config configs/openfungi_stage1_micro.toml --set modality=colony --set num_workers=4 --set amp=true
ls -la runs/stage1/            # of_micro_<arch>.pt, of_macro_<arch>.pt and their provenance JSON
```

Edit `configs/openfungi_stage1_micro.toml` first so its `arch` and recipe match the winner of P1 to P5, and commit that edit.

### 9.6 Tables, explain panels and the blinded review

```bash
source /project/wachiranun.sir/mycoscan/env.sh && cd "$REPO"
mycoscan results runs --out runs/plan1_table.csv                        # light; fine on the login node
mycoscan compare runs/openfungi_macro_well_supported runs/openfungi_macro_ws_convnext_in22k
G="--partition=gpu --gpus=1 --cpus-per-task=4 --mem=32G --time=01:00:00"
sbatch $G --job-name=xai-review "$PROJ"/jobs/cpu.sbatch mycoscan explain \
    --checkpoint runs/openfungi_macro_well_supported/model.pt --manifest data/openfungi/manifest.csv \
    --splits-file data/openfungi/splits_v1.csv --pool B --per-class 5 --seed 0 --out runs/xai_review
```

`runs/xai_review/review_sheet.csv` and the panel PNGs go to the two raters; `review_key.csv` stays with you. Score the returned sheets with `mycoscan score-review` on the login node.

## 10. Plan 2: CMU main study

1. **Upload** `data/cmu/manifest.csv` and the images it points to into the clone (section 5), and the sealed `splits_cmu.csv` with its `.sha256` if sealing happened elsewhere. Otherwise the sealer runs `mycoscan seal` on the login node; it is deterministic and light.
2. **Point the Stage-2 configs at the Stage-1 checkpoints.** In `configs/cmu_microscopic.toml` set `weights = "runs/stage1/of_micro_<arch>.pt"`, the matching `arch`, `splits_file = "data/cmu/splits_cmu.csv"` and `seeds = [0, 1, 2]`; same for `cmu_colony.toml` with the macro checkpoint. Commit.
3. **Sequential and direct arms, paired.**

   ```bash
   cd /project/wachiranun.sir/mycoscan
   s=$(sbatch --parsable --job-name=m1-micro-seq --time=72:00:00 jobs/train.sbatch configs/cmu_microscopic.toml \
         --set run_name=cmu_micro_convnext_sequential)
   d=$(sbatch --parsable --job-name=m1-micro-direct --time=72:00:00 jobs/train.sbatch configs/cmu_microscopic.toml \
         --set weights=imagenet --set run_name=cmu_micro_convnext_direct)
   sbatch --dependency=afterok:$s:$d --job-name=m1-paired jobs/cpu.sbatch \
         mycoscan paired runs/cmu_micro_convnext_sequential runs/cmu_micro_convnext_direct --out runs/m1_paired.json
   ```

4. **Colony arm and late fusion** follow the same pattern with `cmu_colony.toml`, then `mycoscan fuse --colony runs/cmu_colony_... --micro runs/cmu_micro_... --out runs/cmu_fused` as a CPU job (the MLP method embeds images and wants a GPU; add the GPU flags from section 9.5).
5. **The sealed test set, once.** `mycoscan eval --checkpoint runs/<final>/model.pt --manifest data/cmu/manifest.csv --out runs/eval_micro_test` applies the stored tau and never re-tunes it. Run it as a GPU job only after the development-set analysis is frozen.

> **Leakage guards are on.** A training loader that meets a `split = test` row or a Pool B row stops with an error naming the row. If a Plan 2 job dies that way, the manifest or splits file in the clone is wrong; do not patch around it.

## 11. Monitor jobs

| Need | Command |
|---|---|
| My queue, with reasons | `squeue -u "$USER" -o "%.9i %.16j %.9P %.3t %.11M %.11l %.20R"` |
| Follow a running log | `tail -f /project/wachiranun.sir/mycoscan/logs/<name>_<jobid>.out` |
| Where is my job, how long left | `scontrol show job <jobid> \| grep -E "JobState\|RunTime\|TimeLimit\|NodeList\|StdOut"` |
| GPU use on the node it landed on | `nvtopc1` … `nvtopc4` (node number from `NodeList=computeN`); Ctrl-C exits |
| Finished jobs, exit state and peak memory | `sacct -u "$USER" --starttime today --format=JobID,JobName%20,Partition,State,Elapsed,MaxRSS,ExitCode` |
| Cancel one job, or all of mine | `scancel <jobid>`, `scancel -u "$USER"` |
| Partition load and limits | `sinfo`, `scontrol show partition gpu` |
| Disk and credit | `myquota`, `mycredit` |
| Resource cost of a finished run | `python -c "import json;print(json.load(open('runs/<run>/metrics.json'))['resources'])"` |

Email arrives at the address in `--mail-user` when a job ends or fails. A pending job with reason `Priority` or `Resources` will run; `QOSMaxJobsPerUserLimit` means you are over the 2-running limit; `PartitionTimeLimit` means `--time` exceeds the partition cap and the job will never start. `sbatch` itself refusing with `QOSMaxSubmitJobPerUserLimit` means three of your jobs are already queued, dependency-held ones included; submit through `jobs/submit-after.sh` (section 9.3) or wait for one to finish.

## 12. Bring results back

Each run directory holds small CSV, JSON and PNG files plus checkpoints (`model.pt` and one per fold, tens to hundreds of MB each). Pull the small files often and the checkpoints only for the models you will ship or explain.

Pack the metrics and plots without checkpoints on ERAWAN, then copy one file. Do not pipe tar through PowerShell: Windows PowerShell treats the stream as text and corrupts it.

```bash
# on ERAWAN
source /project/wachiranun.sir/mycoscan/env.sh
tar -cf "$PROJ/runs_$(date +%Y%m%d).tar" --exclude='*.pt' --exclude='feature_cache' -C "$REPO" runs
ls -lh "$PROJ"/runs_*.tar
```

```powershell
# on the laptop (PowerShell)
cd C:\Users\USER\OneDrive\Documents\GitHub\CMU-MYCOSCAN
scp "wachiranun.sir@cmu.ac.th@erawan.cmu.ac.th:/project/wachiranun.sir/mycoscan/runs_*.tar" .
tar -xf (Get-Item runs_*.tar | Select-Object -Last 1).Name
Get-ChildItem runs
```

One deployable model:

```powershell
scp "wachiranun.sir@cmu.ac.th@erawan.cmu.ac.th:/project/wachiranun.sir/mycoscan/CMU-MYCOSCAN/runs/<run>/{model.pt,model_card.json}" .\runs\<run>\
```

Keep a second copy of anything irreplaceable (the frozen splits files, Stage-1 checkpoints, the sealed CMU split) under `/home`, which is the 1 TB archive tier, or on the laptop. `/project` quota is 200 GB and `runs/` grows by several GB per multi-seed experiment; delete `folds/*/model.pt` of superseded runs when space runs short.

## 13. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `mycoscan: command not found` in the .err file | `env.sh` not sourced, or conda env missing | Every script starts with `source /project/wachiranun.sir/mycoscan/env.sh`. Run it interactively and `which mycoscan`. |
| `cuda_available: false` inside a GPU job | CPU-only torch wheel installed, or job landed without `--gpus` | `pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall` on the login node; check `nvidia-smi` in the log. |
| Hugging Face "offline mode" or connection error | Compute node tried to download a weight | Add the `(arch, weights)` pair to the section 6 snippet, run it on the login node, resubmit. |
| `CUDA out of memory` | Batch too large for the backbone or bag size | `--set batch_size=16`; keep `amp=true`. ViT-Base and attention-MIL bags are the usual culprits. |
| State `TIMEOUT` in `sacct` | `--time` too short; no resume exists | Read the per-epoch timing in the log, resubmit with 1.5× the projected total. Lower `epochs` only as a deliberate, committed change. |
| State `OUT_OF_MEMORY` (host RAM) | `--mem` too small, often during bootstrap CIs or linear-probe feature caching | Resubmit with `--mem=128G`; on `gpu-h100` or `mixed` you may also raise `--cpus-per-task`. |
| `sbatch: error: QOSMaxSubmitJobPerUserLimit` at submit time | Three jobs already in `squeue`; jobs held by `--dependency` count | Nothing was submitted. Hand the job to `nohup jobs/submit-after.sh <jobid> ...` (section 9.3), or resubmit after one job ends. |
| Job pending forever, reason `PartitionTimeLimit` | `--time` above the partition cap (168 h gpu, 120 h gpu-h100, 24 h mixed) | `scancel` and resubmit under the cap, or split the sweep. |
| `Disk quota exceeded` | /project at 200 GB | `myquota`; remove `runs/*/folds/*/model.pt` of superseded runs, `data/synthetic`, and the smoke run. |
| Run refuses to start: manifest hash differs from the splits file | Manifest was rebuilt after freezing `splits_v1.csv` | Restore the manifest that matches the sidecar hash (your `$PROJ` copy). Never regenerate a frozen split. |
| `dirty: true` in provenance | Uncommitted edits in the clone | `git status`; commit config changes or revert them before the real runs. |
| Python 3.6 or torch 1.10 shows up | Conda env not active, system modules leaking | `module purge` is in `env.sh`; confirm with `python -c "import sys,torch;print(sys.version,torch.__version__)"`. |
| Plots missing, Tk or display error | No X display on compute nodes | `MPLBACKEND=Agg` is set in `env.sh`; make sure the script sources it before any Python runs. |

## 14. Checklist

- [ ] Branch pushed to GitHub; `git log` on ERAWAN shows the expected commit
- [ ] `/project/wachiranun.sir/mycoscan/` laid out; `env.sh` sources cleanly and `which mycoscan` resolves
- [ ] `openfungi/` unpacked inside the clone; file count and hash checked
- [ ] Pretrained weights prefetched for every `arch` and `weights` in the configs
- [ ] Smoke job `COMPLETED` with `cuda_available: true`
- [ ] `pytest` job passed on the `cpu` partition
- [ ] Manifest built; contact sheets reviewed; `splits_v1.csv` frozen and backed up with its sidecar
- [ ] P0 pair submitted; stage-1 pretrain submitted
- [ ] Results pulled to the laptop; `mycredit` and `myquota` checked

---

Sources: ERAWAN wiki pages Slurm, Sbatch, Slurm Commands, Job Submission Methods, Software Usage Example, Storage, File Transfer and Resource Monitoring at hpc.cmu.ac.th (read 2026-10-08); CMU-MYCOSCAN README, pyproject and configs at commit `5f2a407`. Partition limits and software versions change; `sinfo` and `module avail` on the cluster are authoritative.
