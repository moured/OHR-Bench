<h1 align="center">
    Modified OHR-Bench
</h1>

# What is new 
Note : first place doc embeddings and output from the drive to the OHR-Bench.
- quick start script modified to accept --emb_path ./doc_embeddings/
- the code can load existing embeddings
- outputs and embeddings are shared in drive 
- custom.py is also modified to enclode Hybrid mode and loading embeddings
- you can try the run_retrieval.bat in shell folder if you use windows -> run_retrieval.bat gt bge-m3


# Acknowledgement
The evaluation framework is based on [CRUD](https://github.com/IAAR-Shanghai/CRUD_RAG), thanks so much for this brilliant project.

# Citation
```
@article{zhang2024ocr,
  title={OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation},
  author={Junyuan Zhang and Qintong Zhang and Bin Wang and Linke Ouyang and Zichen Wen and Ying Li and Ka-Ho Chow and Conghui He and Wentao Zhang},
  journal={arXiv preprint arXiv:2412.02592},
  year={2024}
}
```

# Copyright Statement
The PDFs are collected from public online channels and community user contributions. Content that is not allowed for distribution has been removed. The dataset is for research purposes only and not for commercial use. If there are any copyright concerns, please contact OpenDataLab@pjlab.org.cn.
