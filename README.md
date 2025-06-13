<h1 style="text-align: center;">:apple: :pear: FruitNeRF++: A Generalized Multi-Fruit Counting Method Utilizing Contrastive Learning and Neural Radiance Fields :peach: :lemon:</h1>

 Lukas Meyer, Andrei-Timotei Ardelean, Tim Weyrich, Marc Stamminger,
<br>

<p align="center">
<a href="https://meyerls.github.io/fruit_nerfpp">🌐[Project Page]</a>
<a href="https://arxiv.org/abs/2505.19863">📄[Paper]</a>
</p>

<p style="align:justify"><b>Abstract</b>: We introduce FruitNeRF++, a novel fruit-counting approach that combines contrastive learning with neural radiance fields to count fruits from unstructured input photographs of orchards. Our work is based on FruitNeRF, which employs a neural semantic field combined with a fruit-specific clustering approach. The requirement for adaptation for each fruit type limits the applicability of the method, and makes it difficult to use in practice. To lift this limitation, we design a shape-agnostic multi-fruit counting framework, that complements the RGB and semantic data with instance masks predicted by a vision foundation model. The masks are used to encode the identity of each fruit as instance embeddings into a neural instance field. By volumetrically sampling the neural fields, we extract a point cloud embedded with the instance features, which can be clustered in a fruit-agnostic manner to obtain the fruit count. We evaluate our approach using a synthetic dataset containing apples, plums, lemons, pears, peaches, and mangoes, as well as a real-world benchmark apple dataset. Our results demonstrate that FruitNeRF++ is easier to control and compares favorably to other state-of-the-art methods. </p>

# News
* Code will be released around August 2025
* 15.9.24: [Project Page](https://meyerls.github.io/fruit_nerfpp) released
