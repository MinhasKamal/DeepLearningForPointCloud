# A Survey on Deep Learning Architectures for Point Cloud Classification and Segmentation
#### Minhas Kamal, Hiranya Garbha Kumar, Balakrishnan Prabhakaran

[Paper (acm)](https://dl.acm.org/doi/epdf/10.1145/3815180) | [Paper (arxiv)](https://arxiv.org/abs/2605.17131)

We reviewed a decade of advancements in point cloud processing: trace the evolution of the field from its foundational roots to the modern SOTA, analyze how diverse architectures overcome the inherent geometric challenges of 3D data, and map out critical research gaps alongside promising future directions.

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

<div align="center"><img width="1000" height=auto alt="image" alt="3d-representation-modalities" src="https://github.com/user-attachments/assets/6e238769-69b0-496d-9a93-7c5471fe8951" /></div>
<!--img width="1867" height="623" alt="3d-representation-modalities" src="https://github.com/user-attachments/assets/fbb73e33-ec8d-4b00-a6eb-64b6e6f38625" /-->

There are several other widely recognized 3D geometric data representation techniques (shown in above Figure), each designed to fulfill a specific set of requirements.

## Point Cloud Datasets
Compared to the widespread availability and diverse use cases of 2D images, 3D data remains relatively less adopted and presents greater complexity in acquisition. Also, with the rise and widespread usage of social media, the volume of 2D visual content has exploded, driven by the ease of capturing, sharing, and consuming photos and videos using smartphones. This proliferation has led to significant advances in 2D computer vision supported by massive datasets. In contrast, the growth of 3D visual data has not followed a similar trajectory. As a result, large-scale and high-quality 3D datasets are rare and have become a significant bottleneck for training and evaluating machine learning (ML) models.

<div align="center"><img width="1000" height=auto alt="classification of 3D datasets" src="https://github.com/user-attachments/assets/2511be76-7dab-4f94-9d13-2cbebd276f3b" /></div>

Most of the 3D datasets are in the form of RGB-D images, meshes, or point clouds. Point clouds can be directly derived from RGB-D data using methods such as SLAM (Simultaneous Localization and Mapping), or from meshes by sampling points on their surfaces, enabling a unified representation across modalities. Figure above illustrates the classification of 3D datasets from three different points of view and table below offers a comparative overview of popular benchmark datasets for point clouds.

<div align="center"><img width="1000" height=auto alt="benchmark dataset table" src="https://github.com/user-attachments/assets/31fc5d50-40cb-459f-94b2-5bd90918c5ef" /></div>

## Classification and Segmentation Architectures

Classification involves assigning probability values to each of the corresponding classes for a given piece of input. This probability value indicates the likelihood of the input data belonging to a specific class, such as animals, cars, furniture, etc. 

<div align="center"><img width="1000" height=auto alt="taxonomy of commonly used deep learning architectures" src="https://github.com/user-attachments/assets/3774d3ee-c474-45ee-aa5d-99aec2353bd2" /></div>

While classification assigns a class value to the entire point cloud, segmentation aims to provide a more granular understanding by allocating a class label to each individual point in the cloud. Among the different types of segmentation tasks, we shall focus on- semantic segmentation and part segmentation. Semantic segmentation implies partitioning a scene into semantically meaningful regions by providing a semantic label for every point; for example, decomposing a rural landscape into regions labeled as grass, water-body, house, tree, and sky. Part segmentation involves segmenting individual objects into different components; for example, dividing a motorcycle image into wheels, seats, oil tank, engine, etc. 

<div align="center"><img width="1000" height=auto alt="comparison among different architectures" src="https://github.com/user-attachments/assets/bd747bc0-e741-4982-9d3c-8d4222953094" /></div>

## Future Research Directions

Significant challenges remain in 3D shape processing, particularly in achieving the robustness, efficiency, and generalization seen in 2D vision systems. Below, we highlight several emerging and promising research directions that aim to address these limitations:

### Self-Supervised Learning
One of the primary reasons 3D image processing continues to lag behind its 2D counterpart is the limited availability of large-scale, high-quality annotated datasets. While 2D vision has benefited immensely from extensive benchmarks such as ImageNet and COCO, the 3D domain lacks datasets of comparable scale, diversity, and annotation richness. As data procurement is expensive, leveraging unlabeled 3D data through pretraining, such as representation learning, is a promising direction. In this paradigm, models are first trained to capture semantically meaningful features, and later fine-tuned on downstream tasks with limited labeled data. This approach not only enhances generalization but also alleviates the challenges posed by the scarcity of large-scale annotated 3D datasets.

### Efficient Attention Mechanism
Effectively handling outliers, artifacts, and variations in point density remains a significant hurdle for real-world applicability. Transformers have shown great promise in extracting high-level point cloud features while handling real-world challenges, but they prove computation heavy especially for large-scale point clouds. Developing more sophisticated attention mechanisms specifically tailored to the unique characteristics of point clouds could lead to improved performance. Future work will likely focus on sparse attention mechanisms, point-cloud-specific tokenizations, and quantized architectures to reduce memory or compute while maintaining performance.

### Multi-Modal and Multi-Sensor Fusion
Modalities such as 2D images, audio, text, and video are significantly more abundant and accessible than 3D data, primarily due to the widespread proliferation of consumer devices, such as smartphones, webcams, and microphones, and the ubiquity of internet platforms, including social media, digital publishing, and streaming services. In contrast, 3D datasets remain limited due to their reliance on specialized hardware, which is not yet widely integrated into mainstream consumer workflows. This scarcity presents a significant bottleneck in training DL models that generalize well across tasks and domains, ultimately hindering the development of robust and transferable 3D representations. Therefore, exploring the combination of point cloud with other modalities could lead to a more comprehensive understanding of 3D scenes. Networks that jointly process 2D and 3D inputs using cross-attention are a growing trend. Similarly, fusing LiDAR and radar or combining multiple LiDAR scans are open problems. Another direction is Cross-Modal sharing- sharing weights between modalities for leveraging knowledge learned from a different modality for a different task.

### Interaction and Motion Estimation
Understanding and predicting the interaction dynamics and motion trajectories of 3D objects is the next reasonable step in comprehensive 3D scene understanding. This involves not only estimating how an object moves in space but also modeling state transitions, for instance, opening a drawer, closing the lid of a laptop, or sliding a window. Also, the manipulation of deformable or non-rigid objects, such as human body, mattresses, plush toys, or fabric materials, is another area for future research. Additionally, recent research has begun to explore material and texture inference from 3D data, which facilitates higher-level scene understanding tasks such as acoustic prediction. For example, the ability to simulate the sound generated by two wooden surfaces sliding against each other.

### Other Research Directions
There are significant gaps in the standardization of point cloud data compression and encoding, which hinder interoperability and efficient storage. Furthermore, there is a pressing need for more advanced sensor technologies capable of capturing high-fidelity, real-world 3D data in diverse environments. The development of large-scale datasets with rich annotations and broad category coverage is equally critical to enable the training of robust and generalizable models. 

To address data scarcity and annotation costs, recent research in point cloud segmentation has increasingly turned toward few-shot learning paradigms. Another emerging and impactful line of work on point cloud semantic segmentation is hierarchical learning; understanding that a "chair" is also a "furniture" item and has "legs". This introduces a probabilistic representation of class hierarchies and regularization to model the intrinsic multi-granularity of 3D scenes. Additionally, the promising capabilities of recent state-space models need to be further explored for point cloud analysis. 

Advancing end-to-end frameworks for 3D object detection and tracking is similarly crucial, especially as current models continue to struggle with small objects, transparent materials, and reflective surfaces. Finally, point cloud generation remains a nascent area and requires significant advancements to achieve practical utility.



## Citation
```
@ARTICLE{DLForPCD2026Kamal,
    author = {Kamal, Minhas and Kumar, Hiranya Garbha and Prabhakaran, Balakrishnan},
    title = {A Systematic Survey on Deep Learning Architectures for Point Cloud Classification and Segmentation},
    year = {2026},
    month = {May},
    publisher = {Association for Computing Machinery (ACM)},
    journal = {ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)},
    address = {New York, NY, USA},
    issn = {1551-6857},
    doi = {10.1145/3815180}
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



See more of my research works on [GoogleScholar](https://scholar.google.com/citations?user=SZxTaQgAAAAJ). I also write about AI on [Medium](https://medium.com/@minhaskamal).
