
# TensorRT INT8 Practice – Day 3 Notes

Today I spent time learning about **converting FP32 models to INT8** using TensorRT. Here’s what I learned:

---

### 1. **TRT Logger**

When we write:

```python
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
```

* TensorRT needs a **logger** to tell us what’s happening.
* It prints **errors, warnings, and info** during engine building or calibration.
* There are different levels (ERROR, WARNING, INFO, VERBOSE), and I chose `INFO` because it gives useful messages without being too noisy.
* Basically, it’s like a “status monitor” for TensorRT.

---

### 2. **CUDA Initialization**

In PyCUDA, before doing anything on GPU, we need to **initialize CUDA**:

```python
import pycuda.driver as cuda
cuda.init()
device = cuda.Device(0)  # pick GPU 0
ctx = device.make_context()
```

* I learned that `pycuda.autoinit` can do this automatically, but I wanted to **manually control it**.
* Manual init is useful because we can choose the GPU, handle multiple GPUs, and clean up the context when done.
* Without initializing CUDA, memory allocation or copying data to GPU will fail.

---

### 3. **Calibration Cache**

* INT8 models need **calibration** to map FP32 values to INT8.
* TensorRT asks us to implement two functions:

```python
def read_calibration_cache(self):
    return None

def write_calibration_cache(self, cache):
    with open("calib_cache.bin", "wb") as f:
        f.write(cache)
```

* `read_calibration_cache`: If we already calibrated before, TensorRT can reuse the cached values instead of recalibrating.
* `write_calibration_cache`: Saves the calibration data so next time we don’t have to run calibration again.
* This is super useful when working with large datasets — saves time.

---

### 4. **GPU Memory Allocation for Calibration**

```python
self.device_input = cuda.mem_alloc(batch_size * 3 * 640 * 640 * 4)  # float32 size
```

* For INT8 calibration, we need to **copy batches of images to the GPU**.
* Each batch needs memory on the GPU, which we manually allocate using PyCUDA.
* This part really helped me understand how **TensorRT interacts with GPU memory**.

---

### 5. **What I Realized Today**

* TensorRT needs **a logger, CUDA context, and calibration data** to convert FP32 to INT8.
* Calibration is essentially “teaching” the engine the range of activations in your dataset.
* Manual CUDA init gives more control, but `autoinit` is fine for quick experiments.
* The calibration cache is a huge time saver.

---

💡 **Next Steps:**
Tomorrow I want to try **running a full YOLOv8 FP32 model** and convert it to INT8 with real calibration data. Also want to experiment with `batch_size > 1` for calibration and see how memory allocation works on the GPU.

---

