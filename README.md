
# 🖼️ Image Recommender System using CLIP + FAISS

This project is a **text-to-image recommendation system** that suggests the most relevant images based on a user-provided text query. It leverages **OpenCLIP** for embedding generation and **FAISS** for efficient similarity search.

---

## 📌 Features

- Accepts a **natural language text** query as input.
- Recommends the **top-k most similar images** from a dataset.
- Uses **OpenCLIP ViT-B-32** model to generate text and image embeddings.
- Stores embeddings in a **FAISS index** for fast nearest-neighbor search.(using Eucleadian distance in this case, can be changed to cosine similarity also)
- Supports **KaggleHub integration** to fetch image datasets.

---

## 🛠️ Setup Instructions

1. **Clone the repo or upload the notebook.**

2. **Install dependencies:**

```bash
pip install open-clip-torch torchvision faiss-cpu pillow kagglehub
```

3. **Run the notebook in Colab or locally with GPU (recommended).**

4. **Dataset download:**

The notebook uses the **Flickr8k** dataset from Kaggle:

```python
import kagglehub
path = kagglehub.dataset_download("aladdinpersson/flickr8kimagescaptions")
```

---

## 🧠 How it Works

1. **Images are embedded** using OpenCLIP’s ViT model.
2. The **embeddings are indexed** using FAISS.
3. When a **text query is entered**, it is also embedded using CLIP.
4. **FAISS performs nearest-neighbor search** to find the most relevant images.

---

## 🔍 Example

```python
query = "a dog playing in the snow"
recommend_images(query, index, id_to_path)
```

Returns the top-n most relevant image paths with cosine similarity scores.

---

## 📦 Output

Each matched image is saved with a new filename under `output_matches/` and optionally displayed in the notebook.

---

## ✨ Credits

- **OpenCLIP** by LAION
- **FAISS** by Facebook AI
- Dataset: [Flickr8k](https://www.kaggle.com/datasets/aladdinpersson/flickr8kimagescaptions)
