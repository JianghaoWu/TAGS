# ZS-GSVERIFY: Zero-Shot Generalist-Specialist Reasoning with Retrieval and Uncertainty-Aware Verification

<p align="center">
  📖 <a href="https://arxiv.org/abs/2505.18283" target="_blank">[Paper on arXiv]</a>
</p>

We propose **ZS-GSVERIFY**, a lightweight, retrieval-augmented multi-agent reasoning framework that enables generalist-specialist collaboration and answer verification without any finetuning.

The framework is composed of three main stages:

1. **Generalist–Specialist Reasoning Collaboration (GSRC)**:
    - A generalist agent analyzes the input medical question and retrieves similar cases as few-shot examples.
    - Multiple specialist agents from different domains provide diverse perspectives and reasoning steps.

2. **Hierarchical Retrieval Augmentation (HRA)**:
    - Enhances generalist and specialist agents with two-stage retrieval:
        - Stage 1: Retrieve similar historical questions.
        - Stage 2: Retrieve chain-of-thoughts (CoT) as guidance for response generation.

3. **Uncertainty-Aware Answer Aggregation (UAAA)**:
    - A verifier agent evaluates the logical consistency and correctness of reasoning steps.
    - Answers are aggregated based on a weighted voting mechanism using confidence scores from the verifier.

![](TAGS\pics\overview.png)

---

## 🧪 Requirements

Install all required python dependencies:

```bash
pip install -r requirements.txt
```

---

## 📚 Data

We evaluate our ZS-GSVERIFY framework on nine medical benchmark datasets:

- **MedQA**
- **MedMCQA**
- **PubMedQA**
- **MedBullets**
- **MMLU**
- **MMLU-Pro**
- **MedXpert-U**
- **MedXpert-R**
- **MedExQA**

📦 Download the datasets and preprocessed files here:  
👉 *All datasets and preprocessed files are already included in `./data`*

---

## 🚀 Run Experiments

Create a `.env` file and add your OpenAI API key or endpoint information.  
Install the `dotenv` package if you haven't already:

```bash
pip install python-dotenv
```

Example `.env` file content:

```bash
AZURE_ENDPOINT=https://azure-openai-miblab-ncu.openai.azure.com/
AZURE_API_KEY=<your_azure_api_key>
AZURE_API_VERSION=2024-08-01-preview
```

Then run the experiments using:

```bash
bash TAGS/run_experiments_all.sh
```

---

## 📌 Citation

If you find our work useful, please consider citing:

```bibtex
@article{wu2025tags,
  title={TAGS: A Test-Time Generalist-Specialist Framework with Retrieval-Augmented Reasoning and Verification},
  author={Wu, Jianghao and Tang, Feilong and Li, Yulong and Hu, Ming and Xue, Haochen and Jameel, Shoaib and Xie, Yutong and Razzak, Imran},
  journal={arXiv preprint arXiv:2505.18283},
  year={2025}
}
```


---

## 🙏 Acknowledgements

Our implementation is partially based on the [MedAgents Benchmark](https://github.com/gersteinlab/medagents-benchmark).  
We thank the authors for providing high-quality code and datasets.
