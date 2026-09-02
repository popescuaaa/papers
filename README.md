# Papers

A repository for paper reproductions and ML research ideas, organized by conceptual foundations rather than just architectures.



## Other things ( special section )


---

## 🎯 Core Concepts & Foundations

### Probabilistic Models & Energy-Based Learning
Understanding the fundamentals of how models learn to represent probability distributions.

- [ ] **Energy-Based Models** — brief overview
  - [ ] Hopfield Networks — modern perspective
  - [ ] Restricted Boltzmann Machines (RBMs)
  - [ ] Boltzmann Machines
- [ ] **Bayesian Networks** — graphical probabilistic models
- [ ] **Hidden Markov Models (HMM)** — sequential probabilistic models
- [ ] **Gaussian Mixture Models (GMM)** — clustering via probability

### Sequence Modeling & Attention
From RNNs to transformers—the evolution of processing sequential data.

- [ ] **RNN Cells** — foundational recurrent architectures
- [ ] **Seq2Seq Models** — encoder-decoder for sequences
- [ ] **Attention Mechanisms** — learning what to focus on
- [ ] **Transformer Architecture** — the attention-only paradigm
- [ ] **Older RNN Variants with Attention** — bridging models

---

## 🎨 Generative Models

Understanding different approaches to learning and generating data distributions.

### Autoregressive & Latent Variable Models
- [ ] **Autoregressive Models** — predicting one token/pixel at a time
- [ ] **Variational Autoencoders (VAEs)** — learning latent representations
  - [ ] Simple VAE — basic implementation
  - [ ] VQ-VAE — vector-quantized variants
- [ ] **Latent Dirichlet Allocation (LDA)** — topic modeling
- [ ] **Diffusion Models** — denoising for generation
  - [ ] Stable Diffusion — practical diffusion implementation

### Adversarial Training
- [ ] **Generative Adversarial Networks (GANs)** — generator vs discriminator
  - [x] Simple GAN + experiments
  - [ ] Training dynamics & joint optimization
  - [ ] DCGAN — convolutional GANs
  - [ ] WGAN — Wasserstein loss improvements
  - [ ] Pix2Pix — conditional image-to-image
  - [ ] CycleGAN — unpaired image translation

---

## 🖼️ Vision Architectures

### Convolutional Neural Networks — Classification
Evolution of CNN-based image classifiers:

- [ ] **AlexNet** (2012) — [Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
  - The breakthrough that started deep learning
- [ ] **ZFNet** (2013) — [Paper](https://arxiv.org/abs/1311.2901)
  - Visualizing what CNNs learn
- [ ] **VGG16** (2014) — [Paper](https://arxiv.org/abs/1505.06798)
  - Simplicity & depth
- [ ] **GoogLeNet** (2014) — [Paper](https://arxiv.org/abs/1409.4842)
  - Multi-scale feature extraction
- [ ] **ResNet** (2015) — [Paper](https://arxiv.org/abs/1704.06904)
  - Skip connections & ultra-deep networks
- [ ] **Inception** (2015) — [Paper](https://arxiv.org/abs/1512.00567)
  - Refined multi-scale approach
- [ ] **Xception** (2016) — [Paper](https://arxiv.org/abs/1610.02357)
  - Separable convolutions
- [ ] **MobileNet** (2017) — [Paper](https://arxiv.org/abs/1704.04861)
  - Efficient architectures for mobile

### Semantic Segmentation
Dense per-pixel prediction:

- [ ] **FCN** (2014) — [Paper](https://arxiv.org/abs/1411.4038)
  - End-to-end fully convolutional networks
- [ ] **SegNet** (2015) — [Paper](https://arxiv.org/abs/1511.00561)
  - Encoder-decoder with pooling indices
- [ ] **UNet** (2015) — [Paper](https://arxiv.org/abs/1505.04597)
  - Skip connections for medical imaging
- [ ] **PSPNet** (2016) — [Paper](https://arxiv.org/abs/1612.01105)
  - Pyramid pooling module
- [ ] **DeepLab** (2016) — [Paper](https://arxiv.org/abs/1606.00915)
  - Atrous convolution for dense prediction
- [ ] **ICNet** (2017) — [Paper](https://arxiv.org/abs/1704.08545)
  - Real-time segmentation
- [ ] **ENet** (2016) — [Paper](https://arxiv.org/abs/1606.02147)
  - Efficient semantic segmentation

### Object Detection
Localization + classification:

- [ ] **RCNN** (2013) — [Paper](https://arxiv.org/abs/1311.2524)
  - Region-based approach
- [ ] **Fast R-CNN** (2015) — [Paper](https://arxiv.org/abs/1504.08083)
  - End-to-end region learning
- [ ] **Faster R-CNN** (2015) — [Paper](https://arxiv.org/abs/1506.01497)
  - Region proposal networks
- [ ] **SSD** (2015) — [Paper](https://arxiv.org/abs/1512.02325)
  - Single-shot multi-scale detection
- [ ] **YOLO** (2015) — [Paper](https://arxiv.org/abs/1506.02640)
  - Real-time single-shot detection
- [ ] **YOLOv2/YOLO9000** (2016) — [Paper](https://arxiv.org/abs/1612.08242)
  - Multi-scale improvements & multi-dataset training

---

## 🔬 Playground

Experimental ideas, prototypes, and explorations:
- [ ] (Add your crazy ML ideas here)

---

## 📚 Learning Path Recommendations

### For Deep Learning Foundations (Beginner)
1. Start with CNN basics: AlexNet → VGG16 → ResNet
2. Learn recurrence: RNN Cells → Seq2Seq
3. Explore: Attention Mechanisms → Transformers

### For Generative Modeling (Intermediate)
1. Probabilistic foundations: GMM → HMM → Bayesian Networks
2. Latent variables: VAE → Autoregressive Models
3. Adversarial: Simple GAN → DCGAN → WGAN
4. Modern: Diffusion Models → Stable Diffusion

### For Computer Vision Tasks (Intermediate)
- **Classification**: VGG16 → ResNet → MobileNet
- **Detection**: RCNN → Faster R-CNN → YOLO
- **Segmentation**: FCN → UNet → DeepLab

---

## 📊 Legend

- [x] Completed & reproduced
- [ ] Not yet started
- Links point to original paper sources (arXiv/NeurIPS)
