# Knowledge Distillation in Neural Networks

Code for the experimentation of the paper "Distilling the Knowledge in a Neural Network"

## Overview

This repository contains an implementation of knowledge distillation using PyTorch. Knowledge distillation is a model compression technique where a smaller model (student) is trained to mimic a larger pre-trained model (teacher). The implementation demonstrates this concept using custom ResNet models on the CIFAR-10 dataset.

In our implementation:
- A larger "generalist" model is trained on all CIFAR-10 classes
- Smaller "specialist" models are trained on subsets of classes, guided by the generalist model

## Requirements

The code requires the following dependencies, which are listed in `requirements.txt`:

```
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.0
matplotlib>=3.3.0
tqdm>=4.50.0
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/knowledge-distillation.git
cd knowledge-distillation
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open the `Knowledge_Distillation_with_Custom_ResNet_Models.ipynb` notebook in your browser.

3. You can run all cells sequentially by clicking "Cell > Run All" or run individual cells with Shift+Enter.

## Code Structure

The notebook is organized into the following sections:

1. **Introduction**: Overview of knowledge distillation
2. **Model Architecture**: Custom ResNet implementation
3. **Data Loading and Preprocessing**: Setup for CIFAR-10
4. **Training and Evaluation Functions**: Functions for model training and evaluation
5. **Knowledge Distillation Implementation**: Implementation of the distillation loss and training process
6. **Main Training Process**: The complete pipeline
7. **Summary**: Recap of the implementation

## Customization Options

You can modify the following parameters in the notebook:

- **Generalist Model Structure**: Modify `create_generalist_model()` to change the depth and width
- **Specialist Model Structure**: Modify `create_specialist_model()` to change the depth and width
- **Class Groups**: Modify the `class_groups` list in the `main()` function to change which classes each specialist model handles
- **Training Parameters**: Change the number of epochs, learning rate, etc. in the training function calls

## Outputs

The training process will generate:
- Model checkpoints stored in the current directory
- Training progress plots saved as PNG files
- Final model files saved in the `models/` directory

## Citation

If you use this code in your research, please cite the original paper:

```
@article{hinton2015distilling,
  title={Distilling the knowledge in a neural network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal={arXiv preprint arXiv:1503.02531},
  year={2015}
}
```

## License

[MIT License](LICENSE)
