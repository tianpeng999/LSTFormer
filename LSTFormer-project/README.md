# LSTFormer: Low-Light Image Enhancement Network Based on Wavelet Transform and Soft-Thresholding Attention



This is the official repository for the paper "Low-Light Image Enhancement Network Based on Wavelet Transform and Soft-Thresholding Attention" (IMAGE-D-26-01258R1).



\*\[A link to the final paper will be added upon publication.]\*



## 📢Code Release Status

[√] Model Architecture (in `models/`)

[√] Inference Code (`test.py`)

[√] Pre-trained Models (links below)

[√] Environment Configuration (`requirements.txt`)

[ ] Training Code (to be released upon paper acceptance)



## 📦Pre-trained Models



Our pre-trained models are too large for GitHub. You can download them from the following links:



\*   \*\*\[LSTFormer on LOL-v1]:\*\* \[通过网盘分享的文件：v1_best_model.pth
链接: https://pan.baidu.com/s/1AWLDltFO2OXguPdx5ob1sQ 提取码: r34w]



\*\*Action Required:\*\* After downloading, please create a `checkpoints/` directory inside the project folder and place the downloaded `.pth` files into it. The final structure should look like this:

LSTFormer-project/

├── checkpoints/

│ ├── best_model.pth

├── models/

├── test.py

├── dataset/

...



## 🛠️Installation



1\.  \*\*Clone the repository:\*\*

&#x20;   ```bash

&#x20;   git clone https://github.com/tianpeng999/LSTFormer.git

&#x20;   cd LSTFormer

&#x20;   ```

2\.  \*\*Create a virtual environment (recommended):\*\*

&#x20;   ```bash

&#x20;   python -m venv venv

&#x20;   source venv/bin/activate  # On Windows, use `venv\\Scripts\\activate`

&#x20;   ```

3\.  \*\*Install dependencies:\*\*

&#x20;   ```bash

&#x20;   pip install -r requirements.txt

&#x20;   ```



## 🏃How to Test (Inference)



1\.  Place your low-light test images in the `dataset/` directory.

2\.  Make sure you have downloaded the pre-trained models and placed them in the `checkpoints/` directory as instructed above.

3\.  Run the inference script. For example, to use the model trained on LOL-v1:

&#x20;   ```bash

&#x20;   python test.py --model\_path checkpoints/best_model.pth --input\_dir images/ --output\_dir results/

&#x20;   ```

&#x20;   The enhanced images will be saved in the `results/` folder, which will be created automatically.



## 🎓Citation



If you find our work useful for your research, please consider citing our paper.

\*(The BibTeX citation will be provided upon publication.)\*



