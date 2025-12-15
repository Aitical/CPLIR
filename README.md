# CPLIR

> **Paper**: *Beyond Degradation Redundancy: Contrastive Prompt Learning for All-in-One Image Restoration*
> 
> **Author**: [Gang Wu](https://scholar.google.com/citations?user=JSqb7QIAAAAJ), [Junjun Jiang](http://homepage.hit.edu.cn/jiangjunjun)*, [Kui Jiang](https://homepage.hit.edu.cn/jiangkui), [Xianming Liu](http://homepage.hit.edu.cn/xmliu), and [Liqiang Nie](https://liqiangnie.github.io/)


## Overview

### 🔥 What is this about?
**All-in-One Image Restoration (AiOIR)** aims to address multiple degradation tasks with a single model. A popular way is to use **task-aware prompts** (a.k.a. instructions or guidance to the restoration backbone). While existing prompt paradigms suffer from two essential issues:

- **Representation Redundancy**: adaptive prompt representations become overlapping and entangled across tasks  
- **Functional Misalignment**: explicit prompts (e.g., from classifiers) are discriminative for *classification*, not necessarily optimal for *reconstruction*

This repo implements **Contrastive Prompt Learning (CPL)**, a general plug-and-play framework that fixes both.

<div align="center"> 
  <table> 
    <tr> 
      <td align="center"><img src="assets/e2e.png" width="310" height="200">
        <br>Adaptive Prompt</td> 
    <td align="center"><img src="assets/prompt.png" width="310" height="200">
      <br>Explicit Prompt</td> 
    <td align="center"><img src="assets/cpl.png" width="310" height="200">
      <br>Contrastive Prompt Learning</td> 
    </tr> 
  </table> 
</div>


---

### ✨ Key Idea 
CPL improves prompt-task alignment through **two complementary components**:

1) Sparse Prompt Module (SPM) — fight redundancy with principled sparsity. Instead of softly blending many prompts, SPM uses **top-k sparse routing** to activate only the most relevant prompt experts, reducing cross-task confusion, and keeps inference efficient.

2) Contrastive Prompt Regularization (CPR) — align prompts by behavior, not embeddings. Conventional regularization operates on **prompt embeddings**. CPR is different: it regularizes the **restoration outcome**.

- ✅ **Positive**: degraded image + correct prompt 
- ❌ **Negative**: same image + mismatched prompts 

If wrong prompts can still produce good restorations, the model is not truly prompt-controlled—CPR explicitly penalizes that.



### 📊 Results

|Benchmark|Dataset |Pretrained Model| Results|
|---|---|---|---|
|WeatherBench| [Download](https://github.com/guanqiyuan/WeatherBench)|[Model](https://drive.google.com/file/d/1XOwUCThKIKLov2FYWtyUA0AIB3rO7E_m/view?usp=sharing)|[Results](https://drive.google.com/file/d/1kCs5XlJJvhYQuNZFgOeQWXZiR2SU-yJS/view?usp=sharing)|

### 🙏 Acknowledgements

This codebase is built upon and inspired by prior works in AiOIR, including (but not limited to): [PromptIR](https://github.com/va1shn9v/PromptIR), [MioIR](https://github.com/Xiangtaokong/MiOIR), [AdaIR](https://github.com/c-yn/AdaIR), [DRSFormer](https://github.com/cschenxiang/DRSformer). Thanks a lot for their nice sharing.


### 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{Wu2025CPLIR,
  title     = {Beyond Degradation Redundancy: Contrastive Prompt Learning for All-in-One Image Restoration},
  author    = {Wu, Gang and Jiang, Junjun and Jiang, Kui and Liu, Xianming and Nie, Liqiang},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year      = {2025}
}
```



