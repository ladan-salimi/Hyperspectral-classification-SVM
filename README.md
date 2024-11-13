
## How to Use
1. **Set Up**:
   - Ensure the required libraries are installed. Use `pip install -r requirements.txt` to install dependencies if a `requirements.txt` file is available.
   - Place the hyperspectral `.mat` files in the appropriate folder.

2. **Run the Program**:
   - Open the `HyperspectralImageClassifier.py` file.
   - Modify the file paths for `image_path` and `gt_path` as required.

3. **Example Usage**:
   Modify the following section in the code to customize your workflow:
   ```python
   classifier = HyperspectralImageClassifier(
       image_path=r'path_to/Indian_pines_corrected.mat',
       gt_path=r'path_to/Indian_pines_gt.mat'
   )

   classifier.read_data()
   classifier.visualize_bands(n=5, save_path=r'path_to/IP_Bands.png')
   classifier.visualize_ground_truth(save_path=r'path_to/IP_GT.png')
   classifier.extract_pixels(save_path=r'path_to/Dataset.csv')
   classifier.apply_pca(n_components=40, save_path=r'path_to/IP_40_PCA.csv')
   classifier.visualize_pca_bands(m=5, save_path=r'path_to/IP_PCA_Bands.png')
   classifier.train_svm()
   classifier.generate_classification_map(save_path=r'path_to/IP_Classification_Map.png')
