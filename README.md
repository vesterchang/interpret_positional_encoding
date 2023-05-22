# About

- A basic usage of this codebase should be almost the same as the original [fairseq](https://github.com/facebookresearch/fairseq/tree/main/fairseq) framework; please follow the documentations available there for the environment setting and data preprocessing.
- To reprocuce the results in the paper, run the following commands:
  
  ```shell
  # Transformer with sinusoidal APE
  bash sin_pe_ende.sh
  bash sin_pe_enfr.sh
  
  # Transformer with posnet
  bash posnet_ende.sh
  bash posnet_enfr.sh
  ```

Note that the `DATA_PATH` variable in the scripts must be assigned with the path to the preprocessed data on your machine. 

All experiments are run on a single Nvidia RTX 3090 GPU card. 

With FP16, about 8~12 GB GPU memory is required.



# Acknowledgements

This repository is based on [fairseq](https://github.com/facebookresearch/fairseq/tree/main/fairseq), a wonderful and state-of-the-art NMT framework.

We would like thank the contributors of the original codebase. 




