# Technical Documentation: GaRT-DETR Model (RGBT Tracking)

This document describes the architecture and data flow of the **GaRT-DETR** model, a multimodal detector/tracker that combines information from Visible (RGB) and Infrared (IR) sensors using spatial attention mechanisms and iterative refinement.

## 1. Temporal Preprocessing (`preprocess_batch`)

The pipeline begins with input data handling. Since the model operates on videos (sequences), this function organizes the dimensional complexity:

* **Temporal Flattening:** Concatenates all frames from all sequences in the batch into a single large tensor () for efficient backbone processing.
* **Resizing:** Ensures that both sensors (RGB and IR) share the same target resolution (e.g., 224x224).
* **Metadata Preservation:** Stores the original image sizes so that, at the end, the bounding boxes can be converted back to the real pixel coordinates of the original video.

## 2. Fusion Blocks with Gating

The model uses two types of blocks to fuse sensor information, determining "how much" to trust each modality:

### `GatedFusionBlock` (Global Fusion)

* **Mechanism:** Uses Cross-Attention so that the primary branch can query the auxiliary branch.
* **Confidence Gate:** Computes a single scalar value for the entire image. If the IR signal is noisy, the gate closes, reducing the influence of IR on RGB.

### `SpatialGatedFusionBlock` (Spatial Fusion)

* **Mechanism:** Unlike the previous block, this one computes confidence **per region** (token).
* **Focus:** If smoke is present in only one part of the RGB frame, the model may choose to trust the IR modality only in that specific region while preserving RGB information for the remainder of the image.

## 3. The Multimodal Backbone (`RGBTBackbone`)

This is the dual-branch feature extractor:

* **RGB Branch (ResNet18):** Specialized in capturing textures, colors, and fine edges.
* **IR Branch (EfficientNet-B0):** Surgically adapted to accept a single thermal channel. It focuses on heat signatures and shapes that remain detectable under low-light conditions.
* **Multi-Level Extraction:** The backbone extracts both deep features (for semantics) and high-resolution features (for precise localization).
* **Multimodal Memory:** The final output is a fused representation that serves as the "visual memory" for the Transformer.

## 4. Refinement Layer (`RefinementLayer`)

This block implements **Soft ROI-Attention**, one of the main innovations of the codebase:

* **Self-Attention:** Queries (drone proposals) interact with each other to avoid duplicate detections.
* **Gaussian Attention:** Instead of attending to the entire image, attention is multiplied by a bias that forces the model to focus only around the current query position.
* **Progressive Focus:** As the layers progress (0 to 6), the viewing radius (Sigma) decreases, forcing the model to become increasingly precise.

## 5. Core Architecture (`GaRT-DETR`)

The master class that orchestrates the temporal and iterative flow:

### Query Initialization

* The model does not start from scratch. It initializes a **Proportional Grid** of reference points distributed across the image, ensuring that no region is ignored at the beginning.

### Temporal Propagation (Smooth Tracking)

* **Alpha-Blending Mechanism:** The model uses confidence from the previous frame to guide the current one. If the drone was confidently detected at frame `t-1`, the query at frame `t` will start exactly from that position, creating smooth and stable tracking.

### Zoom Mechanism (High-Res Zoom)

* During the middle of the refinement process (layer 2), the model pauses to inspect details.
* **`_make_sampling_grid`:** Generates coordinates to "crop" a high-resolution patch from where the object is likely located.
* **Manual ROI Align:** Using `grid_sample`, the model extracts fine details that the deep backbone lost, reintegrating this information into the query.

### Decoupled Prediction

* The model generates two bounding boxes: one for the Visible modality and another for the IR modality. This enables handling of the **Parallax** effect (when sensors are physically displaced) or situations where the drone is visible in one modality but occluded in the other.

## 6. Sampling Engine (`_make_sampling_grid`)

A geometric utility function that:

1. Takes the bounding box coordinates in `[0, 1]`.
2. Converts them to the PyTorch coordinate space `[-1, 1]`.
3. Applies a **1.5x** scaling factor to ensure that the crop contains contextual information around the object.
4. Creates the sampling grid required for the `F.grid_sample` function.

---

### Data Flow Summary

1. **Input:** Batch of RGBT sequences.
2. **Backbone:** Intelligent sensor fusion with confidence gates.
3. **Transformer Encoder:** Global refinement of the memory representation.
4. **Temporal Loop:** Queries are propagated frame-by-frame with smoothing.
5. **Iterative Refinement:** Queries search for the drone within memory using focused attention and local zoom.
6. **Output:** Coordinates and existence scores for both sensor modalities.
