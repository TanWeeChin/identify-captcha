from pathlib import Path
import numpy as np
import cv2
import pickle
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

MODEL_PATH: str = "model/model.pkl"

class Captcha(object):
    def __init__(self):
        self.model = self._load_model()

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """

        if str(im_path).endswith(".txt"):
            img = self._load_rgb_txt(im_path)
        elif str(im_path).endswith(".jpg"):
            img = self._load_jpg_image(im_path)
        imgs = self._get_cropped_images(img)

        captcha_strs_lst = []

        for image in imgs:
            captcha_str = self._infer_captcha(image)
            captcha_strs_lst.append(captcha_str)
        captcha_strs = ''.join(captcha_strs_lst)
        
        # save captcha_strs to save_path
        with open(save_path, "w") as f:
            f.write(captcha_strs)
    
    def _load_model(self):
        """
        Load the model from the pickle file
        """
        model_path = Path(MODEL_PATH)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def _load_rgb_txt(self, path):
        """
        Load the RGB values from the text file and return a numpy array of shape (n, m, 3)
        args:
            path: .txt image path to load
        """
        with open(path, "r") as f:
            lines = f.readlines()
        n, m = map(int, lines[0].strip().split())
        data = [
            [list(map(int, pixel.split(","))) for pixel in line.strip().split()]
            for line in lines[1:]
        ]
        img = np.array(data, dtype=np.uint8)
        
        return img

    def _load_jpg_image(self, file_path):
        """
        Load a JPG image and convert it to RGB format.
        args:
            file_path: .jpg image path to load
        """
        img = cv2.imread(str(file_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _get_cropped_images(self, img):
        """
        Crop the image into individual character images.
        args:
            img: numpy array of shape (n, m, 3) representing the RGB image
        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, bw_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

        bw_img = (bw_img > 127).astype(int)

        min_col = np.min(bw_img, axis=0)
        consecutive_zero_pairs = self._get_consecutive_zero_pairs(min_col)

        min_row = np.min(bw_img, axis=1)
        consecutive_zero_pairs_row = self._get_consecutive_zero_pairs(min_row)

        start_row, end_row = consecutive_zero_pairs_row[0]

        all_images = []

        for i in range(5):
            start_col, end_col = consecutive_zero_pairs[i]
            
            cropped_img = bw_img[start_row:end_row+1, start_col:end_col+1]

            # If cropped image is not 10x8 then resize it to 10x8
            if cropped_img.shape != (10, 8):
                cropped_img = cv2.resize(cropped_img, (8, 10), interpolation=cv2.INTER_NEAREST)

            all_images.append(cropped_img)
        return all_images
    
    def _get_consecutive_zero_pairs(self, arr):
        """
        Get pairs of consecutive zeros in a 1D array.
        args:
            arr: 1D numpy array
        """
        pairs = []
        start_index = None

        for i in range(len(arr)):
            if arr[i] == 0 and start_index is None:
                start_index = i
            elif arr[i] != 0 and start_index is not None:
                pairs.append((start_index, i - 1))
                start_index = None
        if start_index is not None:
            pairs.append((start_index, len(arr) - 1))

        return pairs
    
    def _infer_captcha(self, character_img):
        """
        Infer the captcha character from the given image.
        args:
            character_img: numpy array of shape (10, 8) representing the binary image of a character
        """
        model = self._load_model()

        correlations = {}
        for key in model:
            correlation = np.corrcoef(character_img.flatten(), model[key].flatten())[0, 1]
            correlations[key] = correlation
        predicted_key = max(correlations, key=correlations.get)

        return predicted_key
    
if __name__ == "__main__":

    # Test the Captcha class with a sample input and output file

    file_number = "100"

    input_folder = Path.cwd() / "data" / "input"
    input_file = input_folder / f"input{file_number}.jpg"

    output_folder = Path.cwd() / "data" / "output_test"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"output{file_number}.txt"

    captcha = Captcha()
    captcha(input_file, output_file)