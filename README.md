
# 1. Table of Contents
- [1. Table of Contents](#1-table-of-contents)
- [2. Project](#2-project)
- [3. Folder Structure](#3-folder-structure)
- [4. Solution Formulation](#4-solution-formulation)
  - [4.1. Solution Design \& Reasoning](#41-solution-design--reasoning)
    - [4.1.1. Preprocessing](#411-preprocessing)
    - [4.1.2. Character Segmentation](#412-character-segmentation)
    - [4.1.3. Ensure that fixed pixel template](#413-ensure-that-fixed-pixel-template)
    - [4.1.4. Template Construction](#414-template-construction)
    - [4.1.5. Prediction](#415-prediction)
- [5. Folders](#5-folders)
  - [5.1. Model](#51-model)
  - [5.2. Notebook](#52-notebook)
  - [5.3. Src](#53-src)
- [6. Creating the Virtual Environment](#6-creating-the-virtual-environment)

# 2. Project

- This project is designed to identify unseen captchas. 
- The captchas generated are quite similar each time:
    - the number of characters remains the same each time  
    - the font and spacing is the same each time  
    - the background and foreground colors and texture, remain largely the same
    - there is no skew in the structure of the characters.  
    - the captcha generator, creates strictly 5-character captchas, and each of the characters is either an upper-case character (A-Z) or a numeral (0-9).

# 3. Folder Structure

```
├── data/
│   ├── input/
│   │   ├── input00.jpg
│   │   ├── input00.txt
│   │   ├── ...
│   │   ├── input24.txt
│   │   └── input100.jpg
│   └── output/
│       ├── output00.txt
│       ├── ...
│       └── output24.txt
│
├── model/
│   └── model.pkl
│
├── notebook/
│   ├── creating_comparison_template.ipynb
│   └── EDA.ipynb
│
├── src/
│   └── captcha.py
│
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

# 4. Solution Formulation

## 4.1. Solution Design & Reasoning

Given the relatively small number of samples and the highly consistent structure of the captchas, I opted for a deterministic image-processing and template-comparison approach. 

By segmenting each character and matching it against representative templates, the individual predictions can be combined to reconstruct the full captcha string efficiently and reliably.

---

### 4.1.1. Preprocessing

- Convert RGB → Grayscale  
- Apply thresholding → Binary (Black & White) image  

This cleanly separates characters from the background due to consistent coloring.

---

### 4.1.2. Character Segmentation

Since spacing is fixed and characters are aligned:

- Compute the **minimum pixel value per column and per row**
- Identify consecutive black-column groups -> 5 groups for the 5 characters
- Identify consecutive black-row group -> 1 group
- Crop each character tightly

---

### 4.1.3. Ensure that fixed pixel template

From EDA, most characters are approximately **10 × 8 pixels**.

- Resize each segmented character to **10×8** for consistent comparison

---

### 4.1.4. Template Construction

- Reserve some of the data to be test data.
- From observing the data, we see that there are 
    - Input: 
        - 26 `.jpg` files (additional `input100.jpg`)
        - 25 `.txt` files
    - Output:
        - 24 `.txt` files (missing `output21.txt`)
- Choose to reserve `input20.jpg`, `input21.jpg`, `input22.jpg`, `input23.jpg` and `input100.jpg`as test data.
- Create `output21.txt` and `output100.txt`
- Extract all characters from training images  
- Store in dictionary where:
    - The **key** is the character label  
    - The **value** is a list of binary images corresponding to that character  

    ```
    {
        'A': [imgA1, imgA2, ...],
        ...
    }
    ```

- Compute **mean pixel value per character class**
- Result: one averaged 10×8 template per character

---

### 4.1.5. Prediction

For a new captcha:

1. Segment into 5 characters  
2. Resize each character image to 10×8 pizels
3. Compute correlation of each character with all character templates [Section 4.1.4](#414-template-construction)
4. Select character with highest correlation  
5. Concatenate the 5 predictions to obtain the final captcha string.

---

# 5. Folders

## 5.1. Model


The `model/` directory contains a `model.pkl` file, which stores a Python dictionary of character templates as described in [Section 4.1.4](#414-template-construction). This dictionary maps each character label (A-Z, 0-9) to a list of binary image arrays representing that character, extracted from the training data. The dictionary is used to compute template matching during prediction (see [Section 4.1.5](#415-prediction)).


## 5.2. Notebook

- The `notebook/creating_comparison_template.ipynb` contains the code to create the comparing template and store it as a pickle file in `model/`.

- The `notebook/EDA.ipynb` contains exploratory data analysis. Open it with Jupyter Notebook or VS Code.

## 5.3. Src


The `src/` directory contains the main logic for CAPTCHA recognition, primarily implemented in `captcha.py`. Key components include:

- **Captcha class:** Handles loading the template model, preprocessing images, segmenting characters, and predicting the CAPTCHA string.
    - Loads the template dictionary from `model/model.pkl`.
    - Supports both `.jpg` and `.txt` image inputs.
    - Segments each CAPTCHA image into 5 characters using fixed spacing and pixel analysis.
    - Resizes each character to 10×8 pixels for template matching.
    - Predicts each character by computing correlation with stored templates.
    - Outputs the predicted string to a specified file.
- **Main script:** Provides an example of running the Captcha class on a sample input and saving the result.
 
 # 6. Creating the Virtual Environment
 
 To set up your Python environment and install dependencies:
 - Using `pip install`
 
    ```
    pip install -r requirements.txt
    ```
 - Using  `uv`
    ```
    uv sync
    ```
