# AI Use Appendix

**Name:** Rohil Utture

## 1. Which parts did I use an assistant for, and which did I complete myself?

I used an AI assistant to create the initial structure for the PyTorch and TensorFlow notebook, draft the CUDA matrix-multiplication code, and help organize the written report. I supplied my SID-based parameters and used them in the experiments. I ran the notebooks and CUDA program myself in Google Colab, collected the actual accuracy and timing results, generated the plots and profiler output, and set up the Git repository.

I also checked that the preprocessing and train/validation/test split were shared across the model comparisons, that the requested seeds and hyperparameters were used, and that the reported numbers matched the executed notebook outputs. I revised the assistant's explanations where they did not match the assignment or lecture material, including changing the autoregressive-model discussion to focus on the classical AR(p) time-series definition instead of deep-learning examples such as GPT or WaveNet.

## 2. What was one specific thing the assistant produced that was wrong?

The first version of the neural-network notebook loaded PyTorch and TensorFlow in the same kernel without accounting for a conflict between their native OpenMP libraries. When I ran it, the notebook kernel crashed instead of completing training. The output I saw was:

```text
DeadKernelError: Kernel died
```

This was not a normal model error, such as an incompatible tensor shape or an invalid Python expression. The entire Python process stopped.

## 3. How did I identify the problem?

The data-loading, preprocessing, visualization, and setup cells ran normally. The failure appeared when the notebook reached the framework training code after both TensorFlow and PyTorch had been loaded into the same process. There was no useful Python traceback; the kernel simply died and had to restart. Because the failure happened at the process level, I suspected a native-library problem rather than a mistake in the network architecture or training loop. I then narrowed it down to both frameworks loading conflicting OpenMP runtimes.

## 4. What did I change, and why does the corrected version work?

I set `KMP_DUPLICATE_LIB_OK=TRUE` before importing either framework and imported TensorFlow before PyTorch. This allowed the two frameworks to coexist in the same notebook process during my Colab run. The environment setting permits the duplicate OpenMP runtime libraries to load instead of terminating the process when the conflict is detected, while the import order makes the native libraries initialize consistently.

After making the change, I restarted the runtime and ran the notebook from the first cell through the final evaluation cells. Both frameworks trained successfully, and I verified that the displayed metrics and plots came from the completed run. This fix addresses the library-loading crash; it does not change the model architecture, data split, or evaluation method.
