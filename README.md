# A Systematic Survey on Deep Learning Architectures for Point Cloud Classification and Segmentation
#### Minhas Kamal, Hiranya Garbha Kumar, Balakrishnan Prabhakaran

Point cloud stands as the most widely adopted format for representing 3D shapes and scenes due to its simplicity and geometric fidelity. However, its inherent unordered and irregular nature, exacerbated by sensor noise and occlusions, introduces unique challenges for machine learning based methodologies. To combat these issues, diverse strategies have been developed, including converting to a format that has orderliness, extracting local geometry, and permutation-invariant or self-attention-based processing. In this paper, our focus is directed towards deep learning models for three fundamental tasks in 3D vision: point cloud classification, part segmentation, and semantic segmentation. We begin by formally defining point cloud data, followed by an in-depth discussion on its structural characteristics. Then, we categorize notable works based on their backbone structure and evaluate their performance on popular benchmarks. Beyond empirical comparison, we offer insights into architectural innovations and limitations. We also outline open challenges and promising future directions for 3D point cloud understanding.

## Point Cloud
Point cloud data depicts the surface geometry of an object as a set of discrete data points defined in Cartesian coordinates. Each point has spatial coordinates and may include additional attributes such as color or normal. Although point clouds can be used for any dimensional surface representation, here, we are only considering 3D space. So, a point cloud can be mathematically represented as a set of 3D points:

$$
\mathcal{P} = \\{ \mathbf{p}_i \in \mathbb{R}^3 \mid i = 1, 2, \dots, N \\}, \quad \mathbf{p}_i = (x_i, y_i, z_i) ^ \top
$$

$\mathbf{p}_i$ represents the 3D coordinates of the $i$-th point in Euclidean space. Additional attributes such as color components $(r_i, g_i, b_i)$ and surface normals $(n_i^x, n_i^y, n_i^z)$ can also be included to the point:

$$
\mathbf{p}_i = (x_i, y_i, z_i, r_i, g_i, b_i, n_i^x, n_i^y, n_i^z, \ldots) ^ \top
$$

A point cloud can be derived by sampling mesh surfaces or by fusing multiple depth maps. As there is no explicit connectivity information among the points, it does not directly represent any surface and is inherently unordered and unstructured. Moreover, real-world point clouds contain lots of artifacts and occlusions. However, a point cloud is simple and is arguably the most popular modality for recording, representing, and processing 3D shapes.

<img width="1890" height="629" alt="image" alt="3d-representation-modalities" src="https://github.com/user-attachments/assets/6e238769-69b0-496d-9a93-7c5471fe8951" />
<!--img width="1867" height="623" alt="3d-representation-modalities" src="https://github.com/user-attachments/assets/fbb73e33-ec8d-4b00-a6eb-64b6e6f38625" /-->

There are several other widely recognized 3D geometric data representation techniques (shown in above Figure), each designed to fulfill a specific set of requirements.

## Citation
```
@ARTICLE{DLForPCD2026Kamal,
    author = {Kamal, Minhas and Kumar, Hiranya Garbha and Prabhakaran, Balakrishnan},
    title = {A Systematic Survey on Deep Learning Architectures for Point Cloud Classification and Segmentation},
    year = {2026},
    month = may,
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    issn = {1551-6857},
    journal = {ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)},
    doi = {https://doi.org/10.1145/3815180}
}
```

```
@ARTICLE{GenAIForPCD2024Kamal,
    author = {Kamal, Minhas and Prabhakaran, Balakrishnan},
    title = {Generative AI for 3-D Point Clouds}, 
    journal = {IEEE MultiMedia}, 
    year = {2024},
    volume = {31},
    number = {2},
    pages = {5-6},
    doi = {10.1109/MMUL.2024.3413395}
}
```
