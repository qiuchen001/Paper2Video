# <img src="assets/logo.png" alt="Logo" width="40"/> Paper2Video


<p align="center">
  <b>Automatic Video Generation from Scientific Papers</b>
</p>

<p align="center">
  <a href="https://zeyu-zhu.github.io/webpage/">Zeyu Zhu</a>,
  <a href="https://qhlin.me/">Kevin Qinghong Lin</a>,
  <a href="https://scholar.google.com/citations?user=h1-3lSoAAAAJ&hl=en">Mike Zheng Shou</a> <br>
  Show Lab @ National University of Singapore
</p>


<p align="center">
  <a href="">📄 Paper</a> &nbsp; | &nbsp;
<!--   <a href="">🤗 Daily Paper</a> &nbsp; | &nbsp; -->
  <a href="https://huggingface.co/datasets/ZaynZhu/Paper2Video">🤗 Dataset</a> &nbsp; | &nbsp;
  <a href="">🌐 Project Website</a> &nbsp; | &nbsp;
  <a href="">💬Twitter</a>
</p>

Below is example of Paper2Video for Paper2Video:

https://github.com/user-attachments/assets/ff58f4d8-8376-4e12-b967-711118adf3c4

---

### Table of Contents
- [🌟 Overview](#-overview)
- [🚀 Quick Start: PaperTalker](#-try-papertalker-for-your-paper-)
  - [1. Requirements](#1-requirements)
  - [2. Configure LLMs](#2-configure-llms)
  - [3. Inference](#3-inference)
- [📊 Evaluation: Paper2Video](#-evaluation-paper2video)
- [🙏 Acknowledgements](#-acknowledgements)
- [📌 Citation](#-citation)

---

## 🌟 Overview
<p align="center">
  <img src="assets/teaser.png" alt="Overview" width="100%">
</p>

This work solves two core problems for academic presentations:

- **Left: How to create a presentation video from a paper?**  
  *PaperTalker* — an agent that integrates **slides**, **subtitling**, **cursor grounding**, **speech synthesis**, and **talking-head video rendering**.

- **Right: How to evaluate a presentation video?**  
  *Paper2Video* — a benchmark with well-designed metrics to evaluate presentation quality.


---

## 🚀 Try PaperTalker for your Paper!
<p align="center">
  <img src="assets/method.png" alt="Approach" width="80%">
</p>

### 1. Requirements
Prepare the environment:
```bash
cd src
conda create -n p2v python=3.10
conda activate p2v
pip install -r requirements.txt
````
Download the dependent code and follow the instructions in **Hallo2** to download the model weight.
```bash
git clone https://github.com/fudan-generative-vision/hallo2.git
git clone https://github.com/Paper2Poster/Paper2Poster.git
```
You need to **prepare the environment separately for talking-head generation** to avoide package conflicts, please refer to  <a href="git clone https://github.com/fudan-generative-vision/hallo2.git">Hallo2</a>. After installing, use `which python` to get the python environment path.
```bash
cd hallo2
conda create -n hallo python=3.10
conda activate hallo
pip install -r requirements.txt
```

### 2. Configure LLMs
Export your **API credentials**:
```bash
export GEMINI_API_KEY="your_gemini_key_here"
export OPENAI_API_KEY="your_openai_key_here"
```
The best practice is to use **GPT4.1** or **Gemini2.5-pro** for both LLM and VLMs. We also support locally deployed open-source model(e.g., QWen), details please referring to <a href="https://github.com/Paper2Poster/Paper2Poster.git">Paper2Poster</a>.

### 3. Inference
The script `pipeline.py` provides an automated pipeline for generating academic presentation videos. It takes **LaTeX paper sources** together with **reference image/audio** as input, and goes through multiple sub-modules (Slides → Subtitles → Speech → Cursor → Talking Head) to produce a complete presentation video. ⚡ The minimum recommended GPU for running this pipeline is **NVIDIA A6000** with 48G.

#### Example Usage

Run the following command to launch a full generation:

```bash
python pipeline.py \
    --model_name_t gpt-4.1 \
    --model_name_v gpt-4.1 \
    --model_name_talking hallo2 \
    --result_dir /path/to/output \
    --paper_latex_root /path/to/latex_proj \
    --ref_img /path/to/ref_img.png \
    --ref_audio /path/to/ref_audio.wav \
    --talking_head_env /path/to/hallo2_env \
    --gpu_list [0,1,2,3,4,5,6,7]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name_t` | `str` | `gpt-4.1` | LLM |
| `--model_name_v` | `str` | `gpt-4.1` | VLM |
| `--model_name_talking` | `str` | `hallo2` | Talking Head model. Currently only **hallo2** is supported |
| `--result_dir` | `str` | `/path/to/output` | Output directory (slides, subtitles, videos, etc.) |
| `--paper_latex_root` | `str` | `/path/to/latex_proj` | Root directory of the LaTeX paper project |
| `--ref_img` | `str` | `/path/to/ref_img.png` | Reference image (must be **square** portrait) |
| `--ref_audio` | `str` | `/path/to/ref_audio.wav` | Reference audio (recommended: ~10s) |
| `--ref_text` | `str` | `None` | Optional reference text (for style guidance for subtitles) |
| `--beamer_templete_prompt` | `str` | `None` | Optional reference text (for style guidance for slides) |
| `--gpu_list` | `list[int]` | `""` | GPU list for parallel execution (used in **cursor generation** and **Talking Head rendering**) |
| `--if_tree_search` | `bool` | `True` | Whether to enable tree search for slide layout refinement |
| `--stage` | `str` | `"[0]"` | Pipeline stages to run (e.g., `[0]` full pipeline, `[1,2,3]` partial stages) |
| `--talking_head_env` | `str` | `/path/to/hallo2_env` | python environment path for talking-head generation |
---

## 📊 Evaluation -- Paper2Video
<p align="center">
  <img src="assets/metrics.png" alt="Metrics" width="100%">
</p>

Unlike natural video generation, academic presentation videos serve a highly specialized role: they are not merely about visual fidelity but about **communicating scholarship**. This makes it difficult to directly apply conventional metrics from video synthesis(e.g., FVD, IS, or CLIP-based similarity). Instead, their value lies in how well they **disseminate research** and **amplify scholarly visibility**.From this perspective, we argue that a high-quality academic presentation video should be judged along two complementary dimensions:
#### For the Audience
- The video is expected to **faithfully convey the paper’s core ideas**.  
- It should remain **accessible to diverse audiences**.  

#### For the Author
- The video should **foreground the authors’ intellectual contribution and identity**.  
- It should **enhance the work’s visibility and impact**.  

To capture these goals, we introduce evaluation metrics specifically designed for academic presentation videos: Meta Similarity, PresentArena, PresentQuiz, IP Memory.

### Run Eval
Prepare the environment:
```bash
cd src/evaluation
conda create -n p2v_e python=3.10
conda activate p2v_e
pip install -r requirements.txt
```
For MetaSimilarity and PresentArena:
```bash
python MetaSim_audio.py --r /path/to/result_dir --g /path/to/gt_dir --s /path/to/save_dir
python MetaSim_content.py --r /path/to/result_dir --g /path/to/gt_dir --s /path/to/save_dir
```
```bash
python PresentArena.py --r /path/to/result_dir --g /path/to/gt_dir --s /path/to/save_dir
```
For **PresentQuiz** 
First generate questions from paper and eval using Gemini:
```bash
cd PresentQuiz
python create_paper_questions.py ----paper_folder /path/to/data
python PresentQuiz.py --r /path/to/result_dir --g /path/to/gt_dir --s /path/to/save_dir
```

For **IP Memory**
First generate question pairs from generated videos and eval using Gemini:
```bash
cd IPMemory
python construct.py
python ip_qa.py
```
See the codes for more details!

👉 Paper2Video Benchmark available at:
[HuggingFace](https://huggingface.co/datasets/ZaynZhu/Paper2Video)

---

## 🙏 Acknowledgements

* The souces of the presentation videos are SlideLive and YouTuBe.
* We thank all the authors who spend a great effort to create presentation videos!
* We thank the authors of [Hallo2](https://github.com/fudan-generative-vision/hallo2.git) and [Paper2Poster](https://github.com/Paper2Poster/Paper2Poster.git) for their open-sourced codes.
* We thank all the **Show Lab @ NUS** members for support!



---

## 📌 Citation

If you find our work useful, please cite:

```bibtex
```
