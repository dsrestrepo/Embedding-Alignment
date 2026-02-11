# Multimodal Data Fusion and Embedding Alignment in VLMs

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains a framework for Multimodal Data Fusion using Vision-Language Models (VLMs) and an Embedding Alignment method. The framework allows you to extract, pre-process, and align embedding data from different sources and modalities using powerful VLMs (like CLIP), making efficient use of resources and data by improving zero-shot or few-shot performance through alignment.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Data](#data)
- [Analysis](#analysis)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This framework leverages state-of-the-art Vision-Language Models (VLMs) to combine multimodal data for enhanced predictions. A key focus is **Embedding Alignment**—a method to align embeddings in the same space to reduce the modality gap, thereby improving data fusion tasks without the need for extensive full-model fine-tuning.

## Setup

### Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.12.7
- Required Python packages (specified in `requirements.txt`)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/dsrestrepo/Foundational-Multimodal-Fusion-Benchmark.git
cd Foundational-Multimodal-Fusion-Benchmark
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API (Optional) key if you'll use GPT as foundational model:

Create a `.env` file in the root directory.

Add your OpenAI API key to the `.env` file:

```makefile
OPENAI_API_KEY=your_api_key_here
```
Make sure you have a valid OpenAI API key to access the language model.

## Data

This project uses 8 datasets. You'll find instructions and code about to extract each dataset in `get_datasets.ipynb`:

### General Datasets:

1. [DAQUAR Dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/visual-turing-challenge#c7057): DAQUAR (Dataset for Question Answering on Real-world images) dataset was created for the purpose of advancing research in visual question answering (VQA). It consists of indoor scene images, each accompanied by sets of questions related to the scene's content. The dataset serves as a benchmark for training and evaluating models in understanding images and answering questions about them.

2. [COCO-QA Dataset](https://www.cs.toronto.edu/~mren/research/imageqa/data/cocoqa/): The COCO-QA (COCO Question-Answering) dataset is designed for the task of visual question-answering. It is a subset of the COCO (Common Objects in Context) dataset, which is a large-scale dataset containing images with object annotations. The COCO-QA dataset extends the COCO dataset by including questions and answers associated with the images. Each image in the COCO-QA dataset is accompanied by a set of questions and corresponding answers.

3. [Fakeddit Dataset](https://fakeddit.netlify.app/): Fakeddit is a large-scale multimodal dataset for fine-grained fake news detection. It consists of over 1 million samples from multiple categories of fake news, including satire, misinformation, and fabricated news. The dataset includes text, images, metadata, and comment data, making it a rich resource for developing and evaluating fake news detection models.

4. [Recipes5k Dataset](http://www.ub.edu/cvub/recipes5k/): The Recipes5k dataset comprises 4,826 recipes featuring images and corresponding ingredient lists, with 3,213 unique ingredients simplified from 1,014 by removing overly-descriptive particles, offering a diverse collection of alternative preparations for each of the 101 food types from Food101, meticulously balanced across training, validation, and test splits. The dataset addresses intra- and inter-class variability, extracted from Yummly with 50 recipes per food type.

### Medical Datasets:

5. [BRSET Dataset](https://physionet.org/content/brazilian-ophthalmological/1.0.0/): The Brazilian Multilabel Ophthalmological Dataset (BRSET) stands as a pioneering initiative aimed at bridging the gap in ophthalmological datasets, particularly for under-represented populations in low and medium-income countries. This comprehensive dataset encompasses 16,266 images from 8,524 Brazilian patients, incorporating a wide array of data points including demographics, anatomical parameters of the macula, optic disc, and vessels, along with quality control metrics such as focus, illumination, image field, and artifacts.

6. [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) : The MNIST: HAM10000 dataset is a large collection of dermatoscopic images from different populations, acquired and stored by the Department of Dermatology at the Medical University of Vienna, Austria. It consists of 10,015 dermatoscopic images which can serve as a training set for academic machine learning purposes in tasks like skin lesion analysis and classification, specifically focusing on the detection of melanoma.

7. [mBRSET Dataset](https://www.nature.com/articles/s41597-025-04627-3) : mBRSET is an mobile version of the BRSET dataset, offering mobile camera photos, improved labels and structure for multilabel ophthalmological classification tasks.

8. [MIMIC CXR](https://physionet.org/content/mimic-cxr/2.0.0/#files-panel) : The MIMIC-CXR (Medical Information Mart for Intensive Care, Chest X-Ray) dataset is a large, publicly available collection of chest radiographs with associated radiology reports. It was developed by the MIT Lab for Computational Physiology and provides an extensive resource for training and evaluating machine learning models in the field of medical imaging, particularly in automated radiograph interpretation and natural language processing for clinical narratives.

9. **Joslin dataset**: 

## Usage

1. **Get the Datasets:**
    - Utilize the `Notebooks/get_datasets.ipynb` notebook to acquire the datasets. Functions and code for extraction and preprocessing are provided.

2. **Extract VLM Embeddings:**
    - Use `Notebooks/generate_vlm_embeddings_parallel.ipynb` or the shell script `jobs/job_vlm_embeddings.sh` to extract embeddings from your datasets using state-of-the-art VLMs.
    - **Supported Models:**
        - **CLIP** (Contrastive Language-Image Pre-Training)
        - **SigLIP** (Sigmoid Loss for Language Image Pre-Training)
        - **MedSigLIP** (Medical-adapted SigLIP)
        - **BioMedCLIP** (Biomedical Vision-Language Foundation Model)
    - This step typically generates CSV files containing text and image embeddings which serve as inputs for the alignment process.

3. **Run Alignment Experiments:**
    - The core alignment and training pipeline is managed via `jobs/run_alignment.sh`.
    - This script calls `scripts/run_alignment.py` to:
        - Load the pre-computed embeddings.
        - Apply embedding alignment techniques (shifting distributions).
        - Train Early and Late Fusion classifiers on the aligned embeddings.
        - Evaluate performance and save metrics.
    
    You can configure the datasets, paths, and training parameters (like multilabel flags) directly in `jobs/run_alignment.sh`.

    ```bash
    sbatch jobs/run_alignment.sh
    ```

    **Key Scripts:**
    - `scripts/run_alignment.py`: Main python entry point for alignment and training. Supports argument parsing for multiple datasets and dynamic configuration (e.g., `--multilabel`).
    - `src/classifiers.py`: Contains the `EarlyFusionModel` and `LateFusionModel` definitions, along with the optimized training loops (including early stopping).
    - `src/alignment.py`: (Deprecated/Merged) Logic primarily resides in `run_alignment.py`.

4. **Analysis:**
   - Results (metrics and plots) are saved to the `Images/Alignment` directory (or your configured output path).


## Contributing
Contributions to this research project are welcome. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or research.
3. Make your changes.
4. Create tests.
5. Submit a pull request.


## License
This project is licensed under the MIT License.


## Contact

For any inquiries or questions regarding this project, please feel free to reach out:

- **Email:** davidres@mit.edu
