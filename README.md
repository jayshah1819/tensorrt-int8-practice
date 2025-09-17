

# TensorRT INT8 Practice – 

Today I spent time learning about **converting FP32 models to INT8 using TensorRT**. Here’s what I learned:

---

### 1. **TRT Logger**

```python
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
```

* TensorRT needs a **logger** to tell us what’s happening.
* It prints **errors, warnings, and info** during engine building or calibration.
* There are different levels (ERROR, WARNING, INFO, VERBOSE). I chose `INFO` because it gives useful messages without being too noisy.
* Basically, it’s like a **status monitor** for TensorRT.

---

### 2. **CUDA Initialization**

```python
import pycuda.driver as cuda
cuda.init()
device = cuda.Device(0)  # pick GPU 0
ctx = device.make_context()
```

* PyCUDA requires a **CUDA context** before allocating memory or copying data to the GPU.
* `pycuda.autoinit` can do this automatically, but I wanted **manual control**.
* Manual init is useful because you can **choose the GPU, handle multiple GPUs**, and clean up the context when done.
* Without this, GPU memory allocation or calibration would fail.

---

### 3. **Calibration Cache**

```python
def read_calibration_cache(self):
    return None

def write_calibration_cache(self, cache):
    with open("calib_cache.bin", "wb") as f:
        f.write(cache)
```

* INT8 models need **calibration** to map FP32 values to INT8.
* `read_calibration_cache`: Reuses cached calibration data if it exists.
* `write_calibration_cache`: Saves calibration data for future runs.
* Super useful when working with large datasets—it **saves time**.

---

### 4. **GPU Memory Allocation for Calibration**

```python
self.device_input = cuda.mem_alloc(batch_size * 3 * 640 * 640 * 4)  # float32 size
```

* For INT8 calibration, we copy **batches of images to the GPU**.
* Each batch needs memory on the GPU, manually allocated.
* This helped me understand **how TensorRT interacts with GPU memory**.

---

### 5. **INT8 Calibration Context (`trt.IInt8EntropyCalibrator2`)**

* You define a class that inherits from:

```python
trt.IInt8EntropyCalibrator2
```

* Acts as a **bridge between your dataset and TensorRT**.
* TensorRT calls your class every time it needs a batch for calibration.

---

### 6. **The `get_batch` method**

```python
def get_batch(self, names):
    if self.current_index + self.batch_size > len(self.data):
        return None
    batch = self.data[self.current_index:self.current_index+self.batch_size].ravel()
    cuda.memcpy_htod(self.device_input, batch)
    self.current_index += self.batch_size
    return [int(self.device_input)]
```

* `names` is a **list of input tensor names**, usually `'images'` or `'input_0'` for YOLO.
* In almost all cases, you **ignore it**. TensorRT only expects a list of device pointers.
* This method is where **data actually flows from CPU → GPU** for calibration.

---

### 7. **TensorRT Builder, Network, and Parser**

**Builder:**

```python
builder = trt.Builder(TRT_LOGGER)
```

* The **builder** is responsible for creating the TensorRT engine.
* It manages settings like **precision mode (FP32, FP16, INT8)** and **workspace memory**.

**Workspace Size:**

```python
builder.max_workspace_size = 1 << 30
```

* TensorRT needs **temporary GPU memory** during optimization and engine building.
* `1 << 30` = 1 GB (1 × 2³⁰ bytes) allocated for workspace.
* Bigger workspace can allow **better optimizations**, but it must fit in GPU memory.

**INT8 Mode:**

```python
builder.int8_mode = True
```

* Enables **INT8 precision** in the engine.
* Requires a calibrator to teach the network the correct scaling.

**Network:**

```python
network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
```

* TensorRT requires a **network definition** that describes the model layers.
* `EXPLICIT_BATCH` flag: the network expects **batch size to be specified at runtime**, not fixed at export.
* The builder uses this network to build an **optimized engine**.

**Parser:**

```python
parser = trt.OnnxParser(network, TRT_LOGGER)
```

* Loads the **ONNX model** into the TensorRT network.
* Converts ONNX layers → TensorRT layers.
* After parsing, the network is ready for optimization and engine building.

**Why we build the engine:**

* The **engine** is a GPU-optimized, serialized version of the network.
* It is ready for **inference at maximum speed**, using FP32, FP16, or INT8 precision.
* Engine building is where TensorRT applies **layer fusion, kernel selection, memory planning, and precision conversion**.

---

### 8. **Challenges I Faced**

* Figuring out **manual CUDA init vs autoinit**—I had to understand context creation and cleanup.
* Understanding **get\_batch** and why `names` exists—it’s confusing at first but mostly ignored.
* Deciding **workspace size**—needed trial and error for large models.
* Parsing ONNX → TensorRT network—sometimes errors happen due to unsupported ops, so I learned to export ONNX with `simplify=True` and correct opset.
* INT8 calibration—needed correct preprocessing (normalization, resize, CHW layout) or engine gave wrong results.

---

### 9. **What I Realized Today**

* TensorRT needs **logger, CUDA context, network, builder, parser, and calibration data** to convert FP32 → INT8.
* Calibration is essentially **teaching the engine activation ranges**.
* Manual CUDA init gives more control, but autoinit is fine for experiments.
* Workspace size and engine building are crucial for performance.
* `get_batch` is where CPU data is sent to GPU for calibration, and `names` is mostly a placeholder.

---

💡 **Next Steps:**

Next Steps / Roadmap

YOLOv8 FP32 → INT8 Conversion
Perform full INT8 conversion of the YOLOv8 model using real video calibration data to evaluate performance and accuracy. Experiment with different batch sizes during calibration and optimize the TensorRT workspace size to assess the impact on inference throughput and memory usage.

Fused Preprocessing Kernel
Develop a single, fused CUDA kernel to eliminate CPU bottlenecks in preprocessing raw 4K video frames. The kernel will perform resizing to model input size (e.g., 640×640), BGR→RGB color conversion, uint8→float32 type conversion, normalization (/255.0), and HWC→CHW tensor layout conversion in a single pass. Benchmark this implementation against sequential library-based preprocessing (e.g., OpenCV-CUDA) targeting at least a 5× speedup.

Real-Time Multi-Model Pipeline
Integrate multiple models in a real-time AI inference pipeline with custom hardware optimization. Focus on efficient GPU memory management, batching strategies, low-latency preprocessing, and INT8 inference. The goal is to combine the fused preprocessing kernel with INT8 calibrated engines to achieve consistent real-time performance.


