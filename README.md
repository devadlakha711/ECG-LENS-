# ECG Image to Diagnosis Pipeline

This project builds an end-to-end pipeline that converts **ECG images → digitized signals → model predictions**.

The system performs the following steps:

1. **Digitize ECG images** into numerical signals
2. **Preprocess the signals**
3. **Convert signals to model input format**
4. **Run a trained deep learning model** to predict ECG abnormalities

The project uses the **PTB-XL dataset** for training and evaluation.

---

# Project Structure

```
project/
│
├── ecg_digitiser/              # ECG digitization library
├── ptbxl/                      # PTB-XL metadata
│   ├── ptbxl_database.csv
│   └── scp_statements.csv
│
├── classifier_wrapper.py
├── convert_to_model.py
├── digitizer_runner.py
├── image_to_prediction.py
├── inspect_hr.py
├── plot_digitised.py
├── predict_from_images.py
├── preprocess.py
├── reorganize.py
├── train_ptbxl_multilabel.py
├── verify_npy.py
│
├── requirements.txt
└── README.md
```

---

# Dataset Setup (PTB-XL)

Download the PTB-XL dataset from PhysioNet:

https://physionet.org/content/ptb-xl/1.0.3/

After downloading, place it inside the project directory:

```
ptbxl/
│
├── records100/
├── records500/
├── ptbxl_database.csv
└── scp_statements.csv
```

⚠️ The dataset is ~8GB and therefore **not included in this repository**.

---

# Environment Setup

It is recommended to run this project inside a **Python virtual environment**.

## 1. Create Virtual Environment

```bash
python -m venv venv
```

## 2. Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

# Install Required Python Libraries

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

---

# Generate `requirements.txt`

If you want to regenerate the dependency file after installing packages:

```bash
pip freeze > requirements.txt
```

This will save all installed Python libraries required to reproduce the environment.

---

# Verify Installation

Check installed packages:

```bash
pip list
```

---

# Important Notes

* The `venv/` folder should **NOT be uploaded to GitHub**.
* Add the following entry to `.gitignore`:

```
venv/
__pycache__/
*.pyc
```

---

# Clone ECG Digitizer Dependency

This project uses the **Open-ECG-Digitizer** library.

Clone it inside the project directory:

```bash
git clone https://github.com/Ahus-AIM/Open-ECG-Digitizer.git ecg_digitiser
```

---

# Apply Required Modifications

After cloning the repository, several files must be modified to integrate with this project.

The modified files and instructions are provided below.

(You will add the file list here.)

Example structure:

```
ecg_digitiser/
   file1.py
   file2.py
   module/
      file3.py
```

Replace the corresponding files with the modified versions in this repository.

---

# Next Steps

After setting up the environment and cloning the digitizer repository, you can proceed to run the ECG digitization and prediction pipeline.


# Clone ECG Digitizer

This project depends on the **Open-ECG-Digitizer** library.

Clone it inside the project directory:

```
git clone https://github.com/Ahus-AIM/Open-ECG-Digitizer.git ecg_digitiser
```

This will create the folder:

```
ecg_digitiser/
```

---

# Apply Required Code Modifications

After cloning the digitizer repository, replace the following files with the modified versions provided in this project.

Modified files:

## 1. Modify `ecg_digitiser/run_digitizer.py`

```python
import os
import sys
import glob
import yaml
import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image

_this_dir = os.path.dirname(__file__)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

try:
    from src.model.inference_wrapper import InferenceWrapper
except Exception as e:
    raise RuntimeError("Cannot import InferenceWrapper.") from e

def _find_config_file():
    cfg_dir = os.path.join(os.path.dirname(__file__), "src", "config")
    cand = []
    for ext in ("*.yml", "*.yaml"):
        cand.extend(sorted(glob.glob(os.path.join(cfg_dir, ext))))
    if not cand:
        raise RuntimeError(f"No yaml config files found")
    for path in cand:
        if any(x in os.path.basename(path).lower() for x in ("inference", "infer", "deploy")):
            return path
    return cand[0]

def _load_cfgnode_from_yaml(path):
    from yacs.config import CfgNode as CN
    return CN(yaml.safe_load(open(path, "r", encoding="utf-8")))

def load_png_file(path):
    img = read_image(path).float() / 255.0
    if img.shape[0] > 3: img = img[:3, :, :]
    return img.unsqueeze(0)

def digitize_image_from_path(seg_weights_path, layout_weights_path, image_path,
                             resample_size=2000, target_num_samples=5000, device=None,
                             layout_substring=None):
    if device is None: device = torch.device("cpu")
    elif isinstance(device, str): device = torch.device(device)

    cfg_path = _find_config_file()
    cfg = _load_cfgnode_from_yaml(cfg_path)
    
    from yacs.config import CfgNode as CN
    if not hasattr(cfg, "MODEL"): cfg.MODEL = CN()
    if not hasattr(cfg.MODEL.KWARGS, "config"): cfg.MODEL.KWARGS.config = CN()
    nested_cfg = cfg.MODEL.KWARGS.config

    for sub in ("SEGMENTATION_MODEL", "LAYOUT_IDENTIFIER", "SIGNAL_EXTRACTOR", 
                "CROPPER", "PIXEL_SIZE_FINDER", "DEWARPER", "PERSPECTIVE_DETECTOR"):
        if not hasattr(nested_cfg, sub): setattr(nested_cfg, sub, CN())

    nested_cfg.SEGMENTATION_MODEL.weight_path = seg_weights_path
    nested_cfg.LAYOUT_IDENTIFIER.unet_weight_path = layout_weights_path

    base = os.path.dirname(__file__)
    def _resolve_path(p):
        if not isinstance(p, str): return p
        if os.path.exists(p): return os.path.abspath(p)
        cand = os.path.join(base, "src", "config", os.path.basename(p))
        if os.path.exists(cand): return cand
        return p

    try:
        if hasattr(nested_cfg, "LAYOUT_IDENTIFIER"):
            li = nested_cfg.LAYOUT_IDENTIFIER
            for attr in ("config_path", "unet_config_path", "unet_weight_path"):
                if hasattr(li, attr): setattr(li, attr, _resolve_path(getattr(li, attr)))
        if hasattr(nested_cfg, "SEGMENTATION_MODEL"):
            nested_cfg.SEGMENTATION_MODEL.weight_path = _resolve_path(nested_cfg.SEGMENTATION_MODEL.weight_path)
    except:
        pass

    wrapper = InferenceWrapper(nested_cfg, device=device, resample_size=resample_size, enable_timing=False)
    wrapper = wrapper.to(device).eval()

    img = load_png_file(image_path)
    
    with torch.no_grad():
        out = wrapper(img.to(device), layout_should_include_substring=layout_substring)

    signal_section = out.get("signal", {})
    sig = None
    for k in ("canonical_lines", "lines", "raw_lines"):
        if signal_section.get(k) is not None:
            sig = signal_section.get(k)
            break

    if sig is None: raise RuntimeError("InferenceWrapper returned no signal lines.")
    if torch.is_tensor(sig): sig = sig.cpu().numpy()

    if sig.shape[0] > sig.shape[1]: 
        sig = sig.T

    sig = sig * 0.001
    for i in range(sig.shape[0]):
        series = pd.Series(sig[i])
        series = series.interpolate(method='linear', limit_direction='both')
        series = series.fillna(0)
        sig[i] = series.to_numpy()

    return sig.astype(np.float32)
```

## 2. Modify `ecg_digitiser/src/model/dewarper.py`

```python
import os
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import torch
from skimage.feature import peak_local_max
from sklearn.neighbors import NearestNeighbors
from torch_tps import ThinPlateSpline

DEBUG = False
# If DEBUG is True, specify a directory to save plots
DEBUG_OUTPUT_DIR = "sandbox"
if DEBUG and not os.path.exists(DEBUG_OUTPUT_DIR):
    os.makedirs(DEBUG_OUTPUT_DIR)


class Dewarper(torch.nn.Module):
    def __init__(
        self,
        min_peak_distance_factor: float = 0.7,
        abs_peak_threshold: float = 0.0,
        direction_norm_threshold: float = 0.95,
        magnitude_threshold: float = 0.95,
        optimizer_lr: float = 1.0,
        optimizer_steps: int = 1000,
        optimizer_lr_decay_rate: float = 0.999,
        max_num_warp_points: int = 75,
    ) -> None:
        """
        Initializes the Dewarper with parameters. The dewarper builds a grid of points based on input feature map.
        The grid is then optimized to align with the expected layout of the input data, i.e. a regular grid.
        """
        # IMPORTANT: call parent constructor so torch.nn.Module internal state (_modules, _parameters) is created
        super().__init__()

        # Algorithm / hyper-parameters
        self.min_peak_distance_factor = min_peak_distance_factor
        self.abs_peak_threshold = abs_peak_threshold
        self.direction_norm_threshold = direction_norm_threshold
        self.magnitude_threshold = magnitude_threshold
        self.optimizer_lr = optimizer_lr
        self.optimizer_steps = optimizer_steps
        self.optimizer_lr_decay_rate = optimizer_lr_decay_rate
        self.max_num_warp_points = max_num_warp_points
        self.nn_neighbors = 5  # Number of neighbors for KNN (including self)
        self.kernel_m = 4  # Grid has 4-fold symmetry

        # Properly initialize attributes that will be set later
        self.grid_probabilities: torch.Tensor | None = None
        self.pixels_per_mm: float | None = None
        self.device: torch.device | None = None
        self.target_grid_size: float | None = None
        self.kernel_size: int | None = None
        self.grid: torch.Tensor | None = None

        self.multidim_kernel: torch.Tensor | None = None
        self.channel: torch.Tensor | None = None
        self.local_maxima = None
        self.final_local_maxima = None
        self.final_edges: list[tuple[int, int]] | None = None
        self.optimized_positions: torch.Tensor | None = None

    def _spherical_harmonic_kernel(self, size: int) -> torch.Tensor:
        """
        Create a directional kernel using the real part of 2D spherical harmonics.

        Args:
            size (int): The size of the square kernel (e.g., 21 for a 21x21 kernel).
                        Must be an odd number for symmetric kernel.

        Returns:
            torch.Tensor: A multi-dimensional kernel tensor.
        """
        assert size % 2 == 1, "Size must be odd for symmetric kernel"
        half = size // 2

        # Use torch.meshgrid and torch operations
        y_coords, x_coords = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32, device=self.device),
            torch.arange(-half, half + 1, dtype=torch.float32, device=self.device),
            indexing="ij",
        )
        r = torch.sqrt(x_coords**2 + y_coords**2) + 1e-6  # add epsilon to avoid div-by-zero
        phi = torch.atan2(y_coords, x_coords)
        basis_fcn = torch.cos(self.kernel_m * phi)  # real spherical harmonic: directional pattern
        basis_fcn *= torch.exp(-(r**2) / (2 * (half / 2) ** 2))  # optional Gaussian envelope

        kernel = basis_fcn

        thetas = [0]  # angles in degrees
        zooms = [1]  # zoom factors
        num_rolls = len(thetas)
        num_zooms = len(zooms)

        xc = torch.linspace(-1, 1, kernel.shape[0], device=self.device)
        x_grid, y_grid = torch.meshgrid(xc, xc, indexing="ij")
        coordinates = torch.stack((x_grid, y_grid), dim=-1).reshape(-1, 2)

        def get_transformation(theta: float, zoom: float) -> torch.Tensor:
            """
            Get 2D transformation matrix for angle theta in degrees and zoom factor.

            Args:
                theta (float): Rotation angle in degrees.
                zoom (float): Zoom factor.

            Returns:
                torch.Tensor: The 2D transformation matrix.
            """
            theta_rad = np.deg2rad(theta)  # np.deg2rad is fine as it's a scalar op
            return (
                torch.tensor(
                    [[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]],
                    dtype=torch.float32,
                    device=self.device,
                )
                * zoom
            )

        multidim_ctr = 0
        multidim_kernel = torch.zeros((num_rolls * num_zooms, kernel.shape[0], kernel.shape[0]), device=self.device)
        for theta in thetas:
            for zoom in zooms:
                transformation = get_transformation(theta, zoom)
                transformed_coordinates = coordinates @ transformation.T
                intermediate = torch.nn.functional.grid_sample(
                    kernel.unsqueeze(0).unsqueeze(0),  # kernel is already a torch.Tensor
                    transformed_coordinates.reshape(1, kernel.shape[0], kernel.shape[0], 2),
                    mode="bilinear",
                    align_corners=True,
                ).squeeze()
                intermediate /= intermediate.max()
                multidim_kernel[multidim_ctr] = intermediate
                multidim_ctr += 1

        return multidim_kernel

    def _perform_convolution(self) -> None:
        """
        Performs 2D convolution on the grid probabilities with the generated kernel
        and finds local maxima.
        """
        self.target_grid_size = 5 * self.pixels_per_mm
        self.kernel_size = int(10 * self.pixels_per_mm)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        self.multidim_kernel = self._spherical_harmonic_kernel(self.kernel_size)

        grid_probs_unsqueezed = self.grid_probabilities.unsqueeze(0).unsqueeze(0).to(self.device)
        k = self.multidim_kernel.unsqueeze(1).float().to(self.device)

        grid_probabilities_conv = torch.nn.functional.conv2d(
            grid_probs_unsqueezed, k, padding=self.multidim_kernel.shape[-1] // 2
        )
        self.channel = grid_probabilities_conv.sum(1).squeeze().cpu()  # Move to CPU for peak_local_max
        self.channel /= self.channel.max()

        self.local_maxima = peak_local_max(
            self.channel.numpy(),  # peak_local_max requires numpy
            min_distance=int(self.target_grid_size * self.min_peak_distance_factor),
            threshold_abs=self.abs_peak_threshold,
        )  # type: ignore

        if DEBUG:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.channel.numpy(), cmap="gray")
            plt.scatter(
                self.local_maxima[:, 1],
                self.local_maxima[:, 0],
                c="red",
                s=9,
                label="Local Maxima",
            )
            plt.title("Grid Probabilities after Convolution and Local Maxima")
            plt.colorbar()
            plt.savefig(os.path.join(DEBUG_OUTPUT_DIR, "convolution_maxima.png"))
            plt.show()

    def _filter_and_graph_nodes(self) -> None:
        """
        Filters local maxima based on directionality and magnitude, then constructs
        a graph, keeping only the largest connected component.
        """
        if self.local_maxima.shape[0] == 0:
            self.final_local_maxima = np.array([])
            self.final_edges = []
            return

        knn = NearestNeighbors(n_neighbors=self.nn_neighbors, algorithm="ball_tree", p=1)
        knn.fit(self.local_maxima)
        _, indices = knn.kneighbors(self.local_maxima)

        direction_norms = []
        magnitudes = []
        for i in range(len(self.local_maxima)):
            center = self.local_maxima[i]
            vectors = self.local_maxima[indices[i][1:]] - center

            # Convert to torch tensor for magnitude calculation
            vectors_t = torch.tensor(vectors, dtype=torch.float32, device=self.device)
            vector_sum_norm = torch.norm(torch.sum(vectors_t, dim=0))
            mean_abs_vector_norm = torch.norm(torch.mean(torch.abs(vectors_t), dim=0))

            magnitude = (
                vector_sum_norm / mean_abs_vector_norm
                if mean_abs_vector_norm != 0
                else torch.tensor(0.0, device=self.device)
            )
            magnitudes.append(magnitude.item())  # Store as float

            # Normalize vectors to calculate cosine similarity
            norm_vectors = vectors_t / torch.norm(vectors_t, dim=1, keepdim=True)
            cos_sim = torch.flatten(norm_vectors @ norm_vectors.T)

            sorted_cos_sim = torch.sort(cos_sim)[0]
            cos_val = torch.prod(sorted_cos_sim[:4:2]).item()  # Store as float

            direction_norms.append(cos_val)

        direction_norms_arr = np.array(direction_norms)
        magnitudes_arr = np.array(magnitudes)

        keep_mask = (direction_norms_arr >= self.direction_norm_threshold) * (magnitudes_arr < self.magnitude_threshold)
        refined_local_maxima = self.local_maxima[keep_mask]

        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(keep_mask)[0])}
        G: nx.Graph[int] = nx.Graph()

        for i, original_idx in enumerate(np.where(keep_mask)[0]):
            for j in indices[original_idx][1:]:  # Skip self
                if keep_mask[j]:
                    G.add_edge(i, idx_map[j])

        if len(G) > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        else:
            self.final_local_maxima = np.array([])
            self.final_edges = []
            return

        final_indices = sorted(G.nodes)
        self.final_local_maxima = refined_local_maxima[final_indices]
        index_remap = {old: new for new, old in enumerate(final_indices)}
        self.final_edges = [(index_remap[u], index_remap[v]) for u, v in G.edges]

        if DEBUG:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.channel.numpy(), cmap="gray")
            for src, dst in self.final_edges:
                plt.plot(
                    [self.final_local_maxima[src, 1], self.final_local_maxima[dst, 1]],
                    [self.final_local_maxima[src, 0], self.final_local_maxima[dst, 0]],
                    c="blue",
                    alpha=0.8,
                )
            plt.scatter(
                self.final_local_maxima[:, 1],
                self.final_local_maxima[:, 0],
                c="green",
                s=15,
            )
            plt.title("Largest Connected Component of Filtered KNN Graph")
            plt.colorbar()
            plt.savefig(os.path.join(DEBUG_OUTPUT_DIR, "filtered_graph.png"))
            plt.show()

    def _calculate_distances(self, positions: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        Calculate distances between nodes based on their positions and edges.

        Args:
            positions (torch.Tensor): [N, 2] tensor of node positions (float).
            edges (torch.Tensor): [E, 2] tensor of edge indices (long).

        Returns:
            torch.Tensor: [E] tensor of distances between connected nodes.
        """
        p1 = positions[edges[:, 0]]
        p2 = positions[edges[:, 1]]
        distances: torch.Tensor = torch.norm(p1 - p2, dim=1)
        return distances

    def _layout_loss(self, positions: torch.Tensor, edges: torch.Tensor, target_distance: float) -> torch.Tensor:
        """
        Calculates the layout loss based on deviations from the target distance.

        Args:
            positions (torch.Tensor): [N, 2] tensor of node positions (float).
            edges (torch.Tensor): [E, 2] tensor of edge indices (long).
            target_distance (float): The desired distance between connected nodes.

        Returns:
            torch.Tensor: The calculated layout loss.
        """
        p1 = positions[edges[:, 0]]
        p2 = positions[edges[:, 1]]
        diff = (p1 - p2).abs()
        max_diff = diff.max(dim=1).values
        min_diff = diff.min(dim=1).values
        loss = ((max_diff - target_distance).pow(2) + min_diff.pow(2)).mean().sqrt()
        return loss

    def _plot_positions(self, pos: torch.Tensor, title: str, filename: str, c: npt.NDArray[Any] | None = None) -> None:
        """
        Plots node positions.

        Args:
            pos (torch.Tensor): Tensor of node positions.
            title (str): Title of the plot.
            filename (str): Filename to save the plot.
            c (np.ndarray | None): Color values for scatter plot, if any.
        """
        pos_cpu = pos.detach().cpu().numpy()
        plt.figure(figsize=(10, 5))
        if c is not None:
            plt.scatter(pos_cpu[:, 1], pos_cpu[:, 0], s=10, c=c, alpha=0.5, cmap="jet")
        else:
            plt.scatter(pos_cpu[:, 1], pos_cpu[:, 0], s=10, color="blue", alpha=0.5)
        plt.title(title)
        plt.xlabel("Column Index")
        plt.ylabel("Vertical Position")
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(DEBUG_OUTPUT_DIR, filename))
        plt.show()

    def _get_node_comfort(
        self,
        positions: torch.Tensor,
        edges_tensor: torch.Tensor,
        mean_dist: float,
    ) -> list[float]:
        """
        Calculates a 'comfort' metric for each node based on neighbor distances.

        Args:
            positions (torch.Tensor): Tensor of node positions.
            edges_tensor (torch.Tensor): Tensor of edges.
            mean_dist (float): The target mean distance between nodes.

        Returns:
            list[float]: A list of comfort values for each node.
        """
        node_comfort = []
        for node_idx in range(positions.shape[0]):
            # Find edges where the current node is the source
            outgoing_edges_mask = edges_tensor[:, 0] == node_idx
            # Find edges where the current node is the destination
            incoming_edges_mask = edges_tensor[:, 1] == node_idx

            # Combine neighbors from both directions
            all_neighbors_indices = torch.cat(
                (edges_tensor[outgoing_edges_mask, 1], edges_tensor[incoming_edges_mask, 0])
            ).unique()  # type: ignore

            if len(all_neighbors_indices) > 0:
                neighbor_positions = positions[all_neighbors_indices]
                distances = torch.norm(neighbor_positions - positions[node_idx], dim=1)
                comfort = (distances - mean_dist).abs().mean().item()
            else:
                comfort = 0.0  # Node with no connections
            node_comfort.append(comfort)
        return node_comfort

    def _decay_lr(self, optimizer: torch.optim.Adam, decay_rate: float) -> None:
        """
        Decays the learning rate of the optimizer.

        Args:
            optimizer (torch.optim.Adam): The optimizer whose learning rate will be decayed.
            decay_rate (float): The decay rate.
        """
        for param_group in optimizer.param_groups:
            param_group["lr"] *= decay_rate

    def _optimize_grid_layout(self) -> None:
        """
        Optimizes the layout of the grid points using gradient descent.
        """
        if self.final_local_maxima.shape[0] == 0:
            self.optimized_positions = torch.tensor([])
            return

        coordinates = torch.from_numpy(self.final_local_maxima.copy()).float().to(self.device)
        edges_tensor = torch.tensor(self.final_edges, dtype=torch.long).to(self.device)

        with torch.enable_grad():  # type: ignore
            positions = torch.nn.Parameter(coordinates, requires_grad=True).to(self.device)
            optimizer = torch.optim.Adam([positions], lr=self.optimizer_lr)

            if DEBUG:
                self._plot_positions(positions, "Initial Positions of Nodes", "initial_positions.png")

            for step in range(self.optimizer_steps):
                optimizer.zero_grad()
                loss = self._layout_loss(positions, edges_tensor, target_distance=self.target_grid_size)
                loss.backward()  # type: ignore
                optimizer.step()
                self._decay_lr(optimizer, decay_rate=self.optimizer_lr_decay_rate)

            self.optimized_positions = positions.detach().cpu()

        if DEBUG:
            node_comfort = self._get_node_comfort(positions, edges_tensor, self.target_grid_size)
            self._plot_positions(
                positions,
                "Final Positions of Nodes after Optimization",
                "final_optimized_positions.png",
                c=np.array(node_comfort),
            )

    def _fit_warp(self, device: torch.device = torch.device("cpu")) -> None:
        """
        Warps the original grid probabilities image to the optimized grid layout
        using Thin Plate Spline (TPS).

        Returns:
            torch.Tensor: The warped image.
        """

        input_ctrl = torch.tensor(self.final_local_maxima, dtype=torch.float32)
        output_ctrl = self.optimized_positions

        height, width = self.grid_probabilities.shape
        size = torch.tensor([height, width], dtype=torch.float32).to(device)

        tps = ThinPlateSpline(1, device=device, order=1)

        # Sample control points for TPS if there are too many
        indices = torch.randperm(input_ctrl.shape[0])[: self.max_num_warp_points]
        sampled_input_ctrl = input_ctrl[indices]
        sampled_output_ctrl = output_ctrl[indices]
        try:
            tps.fit(sampled_output_ctrl, sampled_input_ctrl)
        except Exception as e:
            print(f"Error fitting TPS: {e}. The dewarping failed.")
            tps.fit(
                torch.tensor([[1.0, 2.0], [1.0, 3.0]]), torch.tensor([[1.0, 2.0], [1.0, 3.0]])
            )  # Fallback to identity transform

        i = torch.arange(height, dtype=torch.float32)
        j = torch.arange(width, dtype=torch.float32)

        ii, jj = torch.meshgrid(i, j, indexing="ij")
        output_indices = torch.stack((ii, jj), dim=-1).reshape(-1, 2)
        if self.final_local_maxima.shape[0] == 0 or self.optimized_positions.shape[0] == 0:

            self.grid = output_indices.reshape(height, width, 2).to(device)

        input_indices = tps.transform(output_indices).reshape(height, width, 2).to(device)

        grid = 2 * input_indices / size - 1
        self.grid = torch.flip(grid, (-1,)).to(device)

    def transform(self, feature_map: torch.Tensor) -> torch.Tensor:
        if self.grid is None:
            raise ValueError("Grid has not been initialized. Call fit() first.")
        warped = torch.nn.functional.grid_sample(
            feature_map.unsqueeze(0).unsqueeze(0), self.grid[None, ...].to(feature_map.device), align_corners=False
        )[0]
        if DEBUG:
            plt.figure(figsize=(10, 5))
            plt.imshow(feature_map.cpu().numpy(), cmap="gray")
            plt.savefig(os.path.join(DEBUG_OUTPUT_DIR, "original_image.png"))
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.imshow(warped.permute(1, 2, 0).cpu().squeeze().numpy(), cmap="gray")
            plt.savefig(os.path.join(DEBUG_OUTPUT_DIR, "warped_image.png"))
            plt.show()

        return warped.squeeze()

    def fit(self, grid_probabilities: torch.Tensor, pixels_per_mm: float) -> None:
        """
        Executes the full grid processing pipeline for a given image.

        Args:
            grid_probabilities (torch.Tensor): The input grid probabilities tensor of shape (H, W).
            pixels_per_mm (float): Pixels per millimeter, used for kernel and target grid size calculations.

        Returns:
            torch.Tensor: The final warped grid probabilities image on CPU.
        """
        self.grid_probabilities = grid_probabilities
        self.pixels_per_mm = pixels_per_mm
        self.device = grid_probabilities.device

        self._perform_convolution()
        self._filter_and_graph_nodes()
        self._optimize_grid_layout()
        self._fit_warp()
```
## 3. Modify `ecg_digitiser/src/model/inference_wrapper.py`

```python
import time
from contextlib import contextmanager
from typing import Any, Generator

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.nn import Module
from yacs.config import CfgNode as CN

from src.utils import import_class_from_path


@contextmanager
def timed_section(name: str, times_dict: dict[str, float]) -> Generator[None, None, None]:
    """Context manager for timing code blocks.

    Args:
        name: Name of the section.
        times_dict: Dictionary to store timing.
    """
    start = time.time()
    yield
    times_dict[name] = time.time() - start


class InferenceWrapper(Module):
    def __init__(
        self,
        config: CN,
        device: str,
        resample_size: None | tuple[int, ...] = None,
        grid_class: int = 0,
        text_background_class: int = 1,
        signal_class: int = 2,
        background_class: int = 3,
        rotate_on_resample: bool = False,
        enable_timing: bool = False,
        minimum_image_size: int = 512,
        apply_dewarping: bool = True,
    ) -> None:
        """Inference wrapper for ECG pipeline.

        Args:
            config: Configuration node.
            device: Torch device string.
            resample_size: Optional resample target size.
            grid_class: Grid class index.
            text_background_class: Text and background class index.
            signal_class: Signal class index.
            background_class: Background class index.
            rotate_on_resample: Whether to rotate on resample.
            enable_timing: Whether to print timings.
            minimum_image_size: Minimum allowed image size.
            apply_dewarping: Whether to apply dewarping (perspective correction is still performed regardless).
        """
        super().__init__()
        self.config = config
        self.device = device
        self.resample_size = resample_size
        self.grid_class = grid_class
        self.text_background_class = text_background_class
        self.signal_class = signal_class
        self.background_class = background_class
        self.rotate_on_resample = rotate_on_resample
        self._timing_enabled = enable_timing
        self.minimum_image_size = minimum_image_size
        self.apply_dewarping = apply_dewarping

        self.signal_extractor = self._load_signal_extractor()
        self.perspective_detector: Any = self._load_perspective_detector()
        self.segmentation_model: Any = self._load_segmentation_model().to(self.device)
        self.cropper: Any = self._load_cropper()
        self.pixel_size_finder: Any = self._load_pixel_size_finder()
        self.dewarper: Any = self._load_dewarper()
        self.identifier = self._load_layout_identifier()
        self.times: dict[str, float] = {}

    @torch.no_grad()
    def forward(
        self, image: Tensor, layout_should_include_substring: None | str
    ) -> dict[str, Tensor | str | float | None | dict[str, Any]]:
        """Performs full inference on an input image.

        Args:
            image: Input image tensor.
            layout_should_include_substring: Optional substring to filter layout names.

        Returns:
            Dictionary with processed outputs and intermediate results.
        """
        self._check_image_dimensions(image)
        image = self.min_max_normalize(image)
        image = image.to(self.device)

        self.times = {}
        image = self._resample_image(image)

        signal_prob, grid_prob, text_prob = self._get_feature_maps(image)

        with timed_section("Perspective detection", self.times):
            alignment_params = self.perspective_detector(grid_prob)

        with timed_section("Cropping", self.times):
            source_points = self.cropper(signal_prob, alignment_params)

        aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob = self._align_feature_maps(
            image, signal_prob, grid_prob, text_prob, source_points
        )

        with timed_section("Pixel size search", self.times):
            mm_per_pixel_x, mm_per_pixel_y = self.pixel_size_finder(aligned_grid_prob)
            avg_pixel_per_mm = (1 / mm_per_pixel_x + 1 / mm_per_pixel_y) / 2

        with timed_section("Dewarping", self.times):
            if self.apply_dewarping:
                self.dewarper.fit(aligned_grid_prob.squeeze(), avg_pixel_per_mm)
                aligned_signal_prob = self.dewarper.transform(aligned_signal_prob.squeeze())

        with timed_section("Signal extraction", self.times):
            signals = self.signal_extractor(aligned_signal_prob.squeeze())

        self._print_profiling_results()

        layout = self.identifier(
            signals,
            aligned_text_prob,
            avg_pixel_per_mm,
            layout_should_include_substring=layout_should_include_substring,
        )
        try:
            layout_str = layout["layout"]
            layout_is_flipped = str(layout["flip"])
            layout_cost = layout.get("cost", 1.0)
        except KeyError:
            layout_str = "Unknown layout"
            layout_is_flipped = "False"
            layout_cost = 1.0

        return {
            "layout_name": layout_str,
            "input_image": image.cpu(),
            "aligned": {
                "image": aligned_image.cpu(),
                "signal_prob": aligned_signal_prob.cpu(),
                "grid_prob": aligned_grid_prob.cpu(),
                "text_prob": aligned_text_prob.cpu(),
            },
            "signal": {
                "raw_lines": signals.cpu(),
                "canonical_lines": layout.get("canonical_lines", None),
                "lines": layout.get("lines", None),
                "layout_matching_cost": layout_cost,
                "layout_is_flipped": layout_is_flipped,
            },
            "pixel_spacing_mm": {
                "x": mm_per_pixel_x,
                "y": mm_per_pixel_y,
                "average_pixel_per_mm": avg_pixel_per_mm,
            },
            "source_points": source_points.cpu(),
        }

    def _align_feature_maps(
        self,
        image: Tensor,
        signal_prob: Tensor,
        grid_prob: Tensor,
        text_prob: Tensor,
        source_points: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Aligns image and feature maps using perspective cropping.

        Returns:
            Aligned image, signal, grid, and text tensors.
        """
        with timed_section("Feature map resampling", self.times):
            aligned_signal_prob = self.cropper.apply_perspective(signal_prob, source_points, fill_value=0)
            aligned_image = self.cropper.apply_perspective(image, source_points, fill_value=0)
            aligned_grid_prob = self.cropper.apply_perspective(grid_prob, source_points, fill_value=0)
            aligned_text_prob = self.cropper.apply_perspective(text_prob, source_points, fill_value=0)
            if self.rotate_on_resample:
                aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob = self._rotate_on_resample(
                    aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob
                )
            aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob = self._crop_y(
                aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob
            )

            return aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob

    def _crop_y(
        self, image: Tensor, signal_prob: Tensor, grid_prob: Tensor, text_prob: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Crops tensors in y and x using bounds from feature maps.

        Returns:
            Cropped image, signal, grid, and text tensors.
        """

        def get_bounds(tensor: Tensor) -> tuple[int, int]:
            prob = torch.clamp(
                tensor.squeeze().sum(dim=tensor.dim() - 3) - tensor.squeeze().sum(dim=tensor.dim() - 3).mean(),
                min=0,
            )
            non_zero = (prob > 0).nonzero(as_tuple=True)[0]
            if non_zero.numel() == 0:
                return 0, tensor.shape[2] - 1
            return int(non_zero[0].item()), int(non_zero[-1].item())

        y1, y2 = get_bounds(signal_prob + grid_prob)

        slices = (slice(None), slice(None), slice(y1, y2 + 1), slice(None))
        return image[slices], signal_prob[slices], grid_prob[slices], text_prob[slices]

    def _print_profiling_results(self) -> None:
        """Prints the timings for each timed section."""
        if not self._timing_enabled:
            return
        print(" Timing results:")
        max_length = max(len(section) for section in self.times.keys())
        for section, duration in self.times.items():
            print(f"    {section:<{max_length+2}}{duration:.2f} s")
        total_time = sum(self.times.values())
        print(f"Total time: {total_time:.2f} s")

    def _rotate_on_resample(
        self,
        aligned_image: Tensor,
        aligned_signal_prob: Tensor,
        aligned_grid_prob: Tensor,
        aligned_text_prob: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Rotates all tensors if height > width.

        Returns:
            Rotated tensors in same order.
        """
        if aligned_image.shape[2] > aligned_image.shape[3]:
            aligned_image = torch.rot90(aligned_image, k=3, dims=(2, 3))
            aligned_signal_prob = torch.rot90(aligned_signal_prob, k=3, dims=(2, 3))
            aligned_grid_prob = torch.rot90(aligned_grid_prob, k=3, dims=(2, 3))
            aligned_text_prob = torch.rot90(aligned_text_prob, k=3, dims=(2, 3))
        return aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob

    def _resample_image(self, image: Tensor) -> Tensor:
        with timed_section("Initial resampling", self.times):
            if self.resample_size is None:
                return image

            height, width = image.shape[2], image.shape[3]
            min_dim = min(height, width)
            max_dim = max(height, width)

            if min_dim < self.minimum_image_size:
                scale: float = self.minimum_image_size / min_dim
                new_size: tuple[int, int] = (int(height * scale), int(width * scale))
                interpolated: Tensor = F.interpolate(image, size=new_size, mode="bilinear", align_corners=False)
                return interpolated

            if isinstance(self.resample_size, int):
                if max_dim > self.resample_size:
                    scale = self.resample_size / max_dim
                    new_size = (int(height * scale), int(width * scale))
                    return F.interpolate(image, size=new_size, mode="bilinear", align_corners=False, antialias=True)
                return image

            if isinstance(self.resample_size, tuple):
                interpolated = F.interpolate(
                    image, size=self.resample_size, mode="bilinear", align_corners=False, antialias=True
                )
                return interpolated

            raise ValueError(f"Invalid resample_size: {self.resample_size}. Expected int or tuple of (height, width).")

    def process_sparse_prob(self, signal_prob: Tensor) -> Tensor:
        # FIX: Do NOT subtract mean. This kills faint signals/grid lines.
        # signal_prob = signal_prob - signal_prob.mean() * 1  <-- DELETED
        
        # Normalize Min-Max only
        val_min = signal_prob.min()
        val_max = signal_prob.max()
        
        # Avoid division by zero
        if val_max - val_min > 1e-8:
            signal_prob = (signal_prob - val_min) / (val_max - val_min)
            
        # Optional: Gamma correction to boost faint signals
        signal_prob = torch.pow(signal_prob, 0.8) 
        
        return signal_prob

    def _get_feature_maps(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        with timed_section("Segmentation", self.times):
            logits = self.segmentation_model(image)
            prob = torch.softmax(logits, dim=1)

            signal_prob = prob[:, [self.signal_class], :, :]
            grid_prob = prob[:, [self.grid_class], :, :]
            text_prob = prob[:, [self.text_background_class], :, :]

            signal_prob = self.process_sparse_prob(signal_prob)
            grid_prob = self.process_sparse_prob(grid_prob)
            text_prob = self.process_sparse_prob(text_prob)

            return signal_prob, grid_prob, text_prob

    def min_max_normalize(self, image: Tensor) -> Tensor:
        return (image - image.min()) / (image.max() - image.min())

    def _load_signal_extractor(self) -> Any:
        signal_extractor_class = import_class_from_path(self.config.SIGNAL_EXTRACTOR.class_path)
        extractor: Any = signal_extractor_class(**self.config.SIGNAL_EXTRACTOR.KWARGS)
        return extractor

    def _load_perspective_detector(self) -> Any:
        perspective_detector_class = import_class_from_path(self.config.PERSPECTIVE_DETECTOR.class_path)
        perspective_detector: Any = perspective_detector_class(**self.config.PERSPECTIVE_DETECTOR.KWARGS)
        return perspective_detector

    def _load_segmentation_model(self) -> Any:
        segmentation_model_class = import_class_from_path(self.config.SEGMENTATION_MODEL.class_path)
        segmentation_model: Any = segmentation_model_class(**self.config.SEGMENTATION_MODEL.KWARGS)
        self._load_segmentation_model_weights(segmentation_model)
        return segmentation_model.eval()

    def _load_cropper(self) -> Any:
        cropper_class = import_class_from_path(self.config.CROPPER.class_path)
        cropper: Any = cropper_class(**self.config.CROPPER.KWARGS)
        return cropper

    def _load_pixel_size_finder(self) -> Any:
        pixel_size_finder_class = import_class_from_path(self.config.PIXEL_SIZE_FINDER.class_path)
        pixel_size_finder: Any = pixel_size_finder_class(**self.config.PIXEL_SIZE_FINDER.KWARGS)
        return pixel_size_finder

    def _load_dewarper(self) -> Any:
        dewarper_class = import_class_from_path(self.config.DEWARPER.class_path)
        dewarper: Any = dewarper_class(**self.config.DEWARPER.KWARGS)
        return dewarper

    def _load_layout_identifier(self) -> Any:
        layouts = yaml.safe_load(open(self.config.LAYOUT_IDENTIFIER.config_path, "r"))
        unet_cfg = yaml.safe_load(open(self.config.LAYOUT_IDENTIFIER.unet_config_path, "r"))
        unet_class = import_class_from_path(unet_cfg["MODEL"]["class_path"])
        unet: torch.nn.Module = unet_class(**unet_cfg["MODEL"]["KWARGS"])
        checkpoint = torch.load(self.config.LAYOUT_IDENTIFIER.unet_weight_path, map_location=self.device)
        checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        unet.load_state_dict(checkpoint)
        unet.eval()

        identifier_class = import_class_from_path(self.config.LAYOUT_IDENTIFIER.class_path)
        identifier: Any = identifier_class(
            layouts=layouts,
            unet=unet,
            **self.config.LAYOUT_IDENTIFIER.KWARGS,
        )
        return identifier

    def _load_segmentation_model_weights(self, segmentation_model: torch.nn.Module) -> None:
        """Loads weights for segmentation model.

        Args:
            segmentation_model: The model to load weights into.
        """
        checkpoint = torch.load(self.config.SEGMENTATION_MODEL.weight_path, weights_only=True, map_location=self.device)
        if isinstance(checkpoint, tuple):
            checkpoint = checkpoint[0]
        checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        segmentation_model.load_state_dict(checkpoint)

    def _check_image_dimensions(self, image: Tensor) -> None:
        """Checks input image dimensions.

        Args:
            image: Image tensor.

        Raises:
            NotImplementedError: If batch or channel dims are incorrect.
        """
        if image.dim() != 4:
            raise NotImplementedError(f"Expected 4 dimensions, got tensor with {image.dim()} dimensions")
        if image.shape[0] != 1:
            raise NotImplementedError(f"Batch processing not supported, got tensor with shape {image.shape}")
        if image.shape[1] != 3:
            raise NotImplementedError(f"Expected 3 channels, got tensor with shape {image.shape}")
```
## 4. Modify `ecg_digitiser\src\config\inference_wrapper.yml`

```yaml
MODEL:
  class_path: 'src.model.inference_wrapper.InferenceWrapper'
  KWARGS:
    config:
      SIGNAL_EXTRACTOR:
        class_path: 'src.model.signal_extractor.SignalExtractor'
        KWARGS: {}

      PERSPECTIVE_DETECTOR:
        class_path: 'src.model.perspective_detector.PerspectiveDetector'
        KWARGS:
          num_thetas: 250

      DEWARPER:
        class_path: 'src.model.dewarper.Dewarper'
        KWARGS:
          abs_peak_threshold: 0.1

      SEGMENTATION_MODEL:
        class_path: 'src.model.unet.UNet'
        weight_path: './weights/unet_weights_07072025.pt'
        KWARGS:
          num_in_channels: 3
          num_out_channels: 4
          dims: [32, 64, 128, 256, 320, 320, 320, 320]
          depth: 2

      CROPPER:
        class_path: 'src.model.cropper.Cropper'
        KWARGS:
          granularity: 80
          percentiles: [0.02, 0.98]
          alpha: 0.85

      PIXEL_SIZE_FINDER:
        class_path: 'src.model.pixel_size_finder.PixelSizeFinder'
        KWARGS:
          min_number_of_grid_lines: 30
          max_number_of_grid_lines: 70
          lower_grid_line_factor: 0.3

      LAYOUT_IDENTIFIER:
        class_path: 'src.model.lead_identifier.LeadIdentifier'
        config_path: 'src/config/lead_layouts_reduced.yml'
        unet_config_path: 'src/config/lead_name_unet.yml'
        unet_weight_path: './weights/lead_name_unet_weights_07072025.pt'
        KWARGS:
          debug: false
          device: 'cpu'
          possibly_flipped: false

    device: 'cpu'
    resample_size: 3000
    rotate_on_resample: true
    enable_timing: false
    apply_dewarping: false

DATA:
  images_path: '/dataset/ecg-digitization/images_and_scans_with_csvs_labeled_2/'
  image_extensions: ['.png', '.jpg', '.jpeg', '.JPG']
  output_path: 'sandbox/inference_output'
  save_mode: 'all'
  layout_should_include_substring: true # setting this to true will restrict layouts with "limb" or "precordial" in their names, based on filenames
```
## 5. Modify `ecg_digitiser\src\config\inference_wrapper_george-moody-2024.yml`

```yaml
MODEL:
  class_path: 'src.model.inference_wrapper.InferenceWrapper'
  KWARGS:
    config:
      SIGNAL_EXTRACTOR:
        class_path: 'src.model.signal_extractor.SignalExtractor'
        KWARGS: {}

      PERSPECTIVE_DETECTOR:
        class_path: 'src.model.perspective_detector.PerspectiveDetector'
        KWARGS:
          num_thetas: 250

      DEWARPER:
        class_path: 'src.model.dewarper.Dewarper'
        KWARGS:
          abs_peak_threshold: 0.1

      SEGMENTATION_MODEL:
        class_path: 'src.model.unet.UNet'
        weight_path: './weights/unet_weights_07072025.pt'
        KWARGS:
          num_in_channels: 3
          num_out_channels: 4
          dims: [32, 64, 128, 256, 320, 320, 320, 320]
          depth: 2

      CROPPER:
        class_path: 'src.model.cropper.Cropper'
        KWARGS:
          granularity: 80
          percentiles: [0.02, 0.98]
          alpha: 0.85

      PIXEL_SIZE_FINDER:
        class_path: 'src.model.pixel_size_finder.PixelSizeFinder'
        KWARGS:
          min_number_of_grid_lines: 30 # these define the expected number of grid lines on the paper (major vertical lines)
          max_number_of_grid_lines: 70
          lower_grid_line_factor: 0.5

      LAYOUT_IDENTIFIER:
        class_path: 'src.model.lead_identifier.LeadIdentifier'
        config_path: 'src/config/lead_layouts_george-moody-2024.yml' # this file defines the set of possible layouts 
        unet_config_path: 'src/config/lead_name_unet.yml'
        unet_weight_path: './weights/lead_name_unet_weights_07072025.pt'
        KWARGS:
          debug: false
          device: 'cpu'
          possibly_flipped: false
          target_num_samples: 10000 # this is (time * desired sample rate)
          required_valid_samples: 2 # This should be roughly the number of expected rows divided by two
          
    device: 'cpu'
    resample_size: 3000 # this can be changed down to 2000 depending on GPU memory
    rotate_on_resample: true
    enable_timing: false
    apply_dewarping: false
DATA:
  images_path: './gm2024' # change this
  image_extensions: ['.png', '.jpg', '.jpeg', '.JPG'] # set this (maybe)
  save_mode: 'all' # one of ['all', 'timeseries_only', 'png_only']
  layout_should_include_substring: false
  output_path: 'sandbox/inference_output_gm2024' # change this 
  clear_output_dir_if_exists: false # delete all files in output_path if it exists, before running inference

```

## 6. Modify `ecg_digitiser\src\config\inference_wrapper_ahus_testset.yml`

```yaml
MODEL:
  class_path: 'src.model.inference_wrapper.InferenceWrapper'
  KWARGS:
    config:
      SIGNAL_EXTRACTOR:
        class_path: 'src.model.signal_extractor.SignalExtractor'
        KWARGS: {}

      PERSPECTIVE_DETECTOR:
        class_path: 'src.model.perspective_detector.PerspectiveDetector'
        KWARGS:
          num_thetas: 250

      DEWARPER:
        class_path: 'src.model.dewarper.Dewarper'
        KWARGS:
          abs_peak_threshold: 0.1

      SEGMENTATION_MODEL:
        class_path: 'src.model.unet.UNet'
        weight_path: './weights/unet_weights_07072025.pt'
        KWARGS:
          num_in_channels: 3
          num_out_channels: 4
          dims: [32, 64, 128, 256, 320, 320, 320, 320]
          depth: 2

      CROPPER:
        class_path: 'src.model.cropper.Cropper'
        KWARGS:
          granularity: 80
          percentiles: [0.02, 0.98]

      PIXEL_SIZE_FINDER:
        class_path: 'src.model.pixel_size_finder.PixelSizeFinder'
        KWARGS:
          min_number_of_grid_lines: 30
          max_number_of_grid_lines: 70
          lower_grid_line_factor: 0.3

      LAYOUT_IDENTIFIER:
        class_path: 'src.model.lead_identifier.LeadIdentifier'
        config_path: 'src/config/lead_layouts_reduced.yml'
        unet_config_path: 'src/config/lead_name_unet.yml'
        unet_weight_path: './weights/lead_name_unet_weights_07072025.pt'
        KWARGS:
          debug: false
          device: 'cpu'
          possibly_flipped: false

    device: 'cpu'
    resample_size: 3000
    rotate_on_resample: true
    enable_timing: false
    apply_dewarping: false

DATA:
  images_path: '/dataset/ecg-digitization/paper_ecg_dataset/'
  image_extensions: ['.png', '.jpg', '.jpeg', '.JPG']
  output_path: 'sandbox/inference_output'
  save_mode: 'all'
  layout_should_include_substring: True


```


Steps:

1. Navigate to the `ecg_digitiser` directory
2. Replace the corresponding files with the modified versions
3. Save the changes

Example:

```
cp modified_file.py ecg_digitiser/path/to/file.py
```

These modifications adapt the digitizer to work with the **ECG image processing pipeline used in this project**.

---

# Pipeline Overview

The workflow is:

```
ECG Image
     ↓
Digitization (Open-ECG-Digitizer)
     ↓
Signal preprocessing
     ↓
Conversion to model input
     ↓
Deep learning model prediction
```

---

# Running the Pipeline

Digitize ECG images:

```
python digitizer_runner.py
```

Convert digitized signals for model input:

```
python convert_to_model.py
```

Run predictions:

```
python predict_from_images.py
```

---

# Training the Model

To train the PTB-XL multilabel classifier:

```
python train_ptbxl_multilabel.py
```

---

# Acknowledgements

* PTB-XL Dataset (PhysioNet)
* Open-ECG-Digitizer
* WFDB library



