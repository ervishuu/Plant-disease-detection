# Agriculture-Web-Application
Getting affected by a disease is very common in plants due to various factors such as fertilizers, cultural practices followed, environmental conditions, etc. These diseases hurt agricultural yield and eventually the economy based on it. 

Any technique or method to overcome this problem and getting a warning before the plants are infected would aid farmers to efficiently cultivate crops or plants, both qualitatively and quantitatively. Thus, disease detection in plants plays a very important role in agriculture.

In this web Application you can check the diseases of maize, potato, tomato, cotton, grapes and also you can check what crops you need to grow in your land by giving a little information about your land here so you can get a good yield by taking the right crop in that land at the right time




![Screenshot (127)](https://user-images.githubusercontent.com/63738852/117561159-e86ba200-b0b1-11eb-9c0b-b193870d8b06.png)






   Deep neural networks has been highly successful in image classification problems. In this Project, we have used CNN that can recognize the plant diseases of Plant images. We have used publicly available datasets. Hence, the problem that we have addressed is a multi class classification problem. We compared different architectures including VGG16, VGG19, ResNet50, InceptionV3, as the backbones for our work. We found that VGG19,InceptionV3 achieves the best result on the test set.

>VGG19 Architecture :

* So in simple language VGG is a deep CNN used to classify images.
A fixed size of (224 * 224) RGB image was given as input to this network which means that the matrix was of shape (224,224,3).

* The only preprocessing that was done is that they subtracted the mean RGB value from each pixel, computed over the whole training set.

* Used kernels of (3 * 3) size with a stride size of 1 pixel, this enabled them to cover the whole notion of the image.

* spatial padding was used to preserve the spatial resolution of the image.

* max pooling was performed over a 2 * 2 pixel windows with sride 2.

* this was followed by Rectified linear unit(ReLu) to introduce non-linearity to make the model classify better and to improve computational time as the previous models used tanh or sigmoid functions this proved much better than those.




![vgg-ispravljeno--718x1024](https://user-images.githubusercontent.com/63738852/117563247-253f9500-b0c2-11eb-9f26-b86360ae5159.png)


