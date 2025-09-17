

# TensorRT INT8 Practice –Notes

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

* `pycuda.autoinit` can do this automatically, but I wanted to **manually control it**.
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
* This helped me understand how **TensorRT interacts with GPU memory**.

---

### 5. **INT8 Calibration Context (`trt.IInt8EntropyCalibrator2`)**

When doing INT8 calibration, you define a class that inherits from:

```python
trt.IInt8EntropyCalibrator2
```

* This class is like a **bridge** between your dataset and TensorRT.
* TensorRT calls your class every time it needs a batch for calibration.

---

### 6. **The `get_batch` method**

```python
def get_batch(self, names):
    # Ignore 'names', just feed your batch
    if self.current_index + self.batch_size > len(self.data):
        return None  # no more batches

    batch = self.data[self.current_index:self.current_index+self.batch_size].ravel()
    cuda.memcpy_htod(self.device_input, batch)
    self.current_index += self.batch_size

    return [int(self.device_input)]
```

**Key points:**

1. `names` is a **list of input tensor names** for the network.

   * TensorRT passes it so you know which input you are supposed to feed.
   * For YOLO models, usually there’s only **one input**, e.g., `'images'` or `'input_0'`.
   * ✅ In almost all cases, you **ignore it**. It’s just required by TensorRT’s function signature.

2. TensorRT only cares that you **return a list of device pointers** matching the input tensors.

---

### 7. **What I Realized Today**

* TensorRT needs **a logger, CUDA context, and calibration data** to convert FP32 to INT8.
* Calibration is essentially “teaching” the engine the range of activations in your dataset.
* Manual CUDA init gives more control, but `autoinit` is fine for quick experiments.
* The calibration cache is a huge time saver.
* The `get_batch` method is where the **data actually flows from CPU to GPU** for calibration, and `names` is mostly just a placeholder in YOLO INT8 calibration.

---

💡 **Next Steps:**
Tomorrow I want to try **running a full YOLOv8 FP32 model** and convert it to INT8 with real calibration data. Also want to experiment with `batch_size > 1` for calibration and see how memory allocation works on the GPU.

