# Image_similarity_search_engine
An interactive image similarity search engine built with PyTorch and k-Nearest Neighbors (k-NN), designed to retrieve visually similar images from a dataset.

🚀 Features
Deep Feature Extraction: Utilizes a pre-trained convolutional neural network (e.g., ResNet) to extract feature embeddings from images.

Similarity Computation: Employs cosine similarity to measure the likeness between image embeddings.

Efficient Retrieval: Implements k-NN algorithm for fast and accurate retrieval of similar images.

🛠️ Installation
Clone the Repository:

bash
Copy
Edit
git clone https://huggingface.co/spaces/junaid17/image_search_engine
cd image_search_engine
Install Dependencies:

bash
Copy
Edit
pip install -r requirements.txt
📂 Project Structure
Embeddings.ipynb: Notebook to generate and save image embeddings.

compute_similarity.ipynb: Notebook to compute similarities between images using embeddings.

embeddings.pkl: Serialized embeddings of the image dataset.

filenames.pkl: Serialized list of image filenames corresponding to embeddings.

requirements.txt: List of required Python packages.

Dataset : https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

README.md: Project documentation.
YouTube
+5
GitHub
+5
arXiv
+5

📈 How It Works
Embedding Generation: Images are passed through a pre-trained CNN to extract feature vectors.

Similarity Computation: Cosine similarity is calculated between the query image's embedding and those in the dataset.

Retrieval: Top-k similar images are retrieved based on similarity scores.


![image/png](https://cdn-uploads.huggingface.co/production/uploads/67bdbcf6136f00a4bd3a1d49/jmoPwNMNowIxcfkpwbKtM.png)
