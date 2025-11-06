#!/bin/bash  
#SBATCH --job-name=diffpure                 # Job name
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ssanjaya@ufl.edu        # Where to send mail
#SBATCH --ntasks=1                          # Run a single task
#SBATCH --mem=32gb                         # Memory limit
#SBATCH --time=48:00:00                     # Time limit hrs:min:sec
#SBATCH --cpus-per-task=4                   # Use 4 cores
#SBATCH --output=diffpure%j.log             # Standard output and error log
#SBATCH --error=diffpure%j.log
#SBATCH --partition=hpg-b200  
#SBATCH --gpus=b200:1  
  
pwd; hostname; date

module purge
module load cuda/12.9.1

# Arch & CUDA env (Blackwell)
export TORCH_CUDA_ARCH_LIST="10.0"

# Ensure CUDA_HOME is set (module usually sets it; if not, infer from nvcc)
: "${CUDA_HOME:=$(dirname "$(dirname "$(which nvcc)")")}"
export CUDA_HOME
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version || true

# Optional: use all CPU threads for dataloaders/OpenMP
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Clean any failed torch extension builds from earlier runs
rm -rf ~/.cache/torch_extensions

source ../diffpureenv/bin/activate

python -V
python -c "import torch,sys;print('torch',torch.__version__,'cuda',torch.version.cuda)"

# Sanity: see the GPU and visibility
nvidia-smi || true
python - <<'PY'
import torch, os
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
PY

python - <<'PY'
import torch
x = torch.randn(1024, 1024, device='cuda')
y = torch.matmul(x, x)  # forces a kernel
print("OK matmul on:", torch.cuda.get_device_name(0), "shape:", y.shape)
print("torch:", torch.__version__, "torch.cuda:", torch.version.cuda)
PY

srun python part1.py

date
