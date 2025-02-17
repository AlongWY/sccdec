# Self-Constructed Context Decompilation with Fined-grained Alignment Enhancement

Code For [Self-Constructed Context Decompilation with Fined-grained Alignment Enhancement](https://aclanthology.org/2024.findings-emnlp.385).

## Deploy

```bash
vllm serve LLM4Binary/llm4decompile-6.7b-v1.5 --port 8000 --tensor-parallel-size 1 --enable-lora --lora-modules model=ylfeng/sccdec-lora
python src/sccdec/eval.py --base_url http://127.0.0.1:8000/v1 --model_name model --one_shot
```

## Resoureces

+ [Code in Github](https://github.com/AlongWY/sccdec)
+ [Model in huggingface](https://huggingface.co/ylfeng/sccdec-lora).
+ [Dataset in huggingface](https://huggingface.co/datasets/ylfeng/sccdec-dataset).

## Reference

If you use SCCDEC in your work, please cite it as follows:

```
@inproceedings{feng-etal-2024-self,
    title = "Self-Constructed Context Decompilation with Fined-grained Alignment Enhancement",
    author = "Feng, Yunlong  and
      Teng, Dechuan  and
      Xu, Yang  and
      Mu, Honglin  and
      Xu, Xiao  and
      Qin, Libo  and
      Zhu, Qingfu  and
      Che, Wanxiang",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.385",
    pages = "6603--6614",
    abstract = "Decompilation transforms compiled code back into a high-level programming language for analysis when source code is unavailable. Previous work has primarily focused on enhancing decompilation performance by increasing the scale of model parameters or training data for pre-training. Based on the characteristics of the decompilation task, we propose two methods: (1) Without fine-tuning, the Self-Constructed Context Decompilation (sc$^2$dec) method recompiles the LLM{'}s decompilation results to construct pairs for in-context learning, helping the model improve decompilation performance. (2) Fine-grained Alignment Enhancement (FAE), which meticulously aligns assembly code with source code at the statement level by leveraging debugging information, is employed during the fine-tuning phase to achieve further improvements in decompilation. By integrating these two methods, we achieved a Re-Executability performance improvement of approximately 3.90{\%} on the Decompile-Eval benchmark, establishing a new state-of-the-art performance of 52.41{\%}. The code, data, and models are available at https://github.com/AlongWY/sccdec.",
}
```
* License: MIT
