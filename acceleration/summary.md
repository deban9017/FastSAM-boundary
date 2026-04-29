# Performance Benchmarking & Optimization Summary

## Acceleration Experiments & Results

We benchmarked a 192KB recurrent **Boundary Puller + Smoother** pipeline against **FastSAM-x** on an Intel i7-12650H CPU. Starting from a raw PyTorch latency of **~1.8s**, we achieved a final optimized latency of **0.85s** (a **2.1x speedup**) through the following steps:

1.  **ONNX Export & Fusion:** Combining the 5-iteration recurrent loop and the Smoother into a single unified computational graph eliminated Python-to-C++ handoff overhead.
2.  **Batching:** Processing patches in groups of 32 utilized CPU vectorization (AVX) better than serial processing.
3.  **Parallelism Testing:** Attempting multi-threading and massive "Big Batch" processing actually **degraded** performance (3.5s+), proving that memory bandwidth and cache limits are the primary bottlenecks for small models.

**Result:** Our model refines boundaries in **<50% of the time** required for the initial FastSAM-x segmentation (**1.76s**), making it a highly efficient post-processing module for edge deployment.

---

## Technical Reasoning: Why the Results Varied

* **Success of Fusion:** Fusing the models kept the tensors in the CPU’s high-speed **L1/L2 cache**. Separated models forced the CPU to dump data to the slower **System RAM** between steps.
* **Failure of Multi-threading:** For a 192KB model, the time the OS takes to "manage" threads (context switching) is longer than the actual math. The overhead exceeded the computation.
* **Failure of "Big Batching":** Attempting to process all 15 patches at once caused **Cache Thrashing**. The data volume exceeded the 24MB L3 cache, forcing the CPU to wait on the narrow memory bus.
* **The 0.85s Ceiling:** This represents the hardware's "balance point" where the CPU's compute cores are fully saturated without being starved for data by the memory controller.