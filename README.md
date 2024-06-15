https://arxiv.org/pdf/1312.6211
We find that in most cases, dropout increases the opti- mal size of the net, so the resistance to forgetting may be explained mostly by the larger nets having greater capacity. However, this effect is not consistent, and when using dissimilar task pairs, dropout usually de- creases the size of the net.



Dataset: MovieLens
Kubeflow pipeline tutorial: <https://www.youtube.com/watch?v=6wWdNg0GMV4&t=2642s>


MEMORY-BASED APPROACHES
Weighted Average (user-user and item-item): <https://www.kaggle.com/code/iambideniz/hybrid-recommender-system/notebook>

KNN (item-item) - once you have the k nearest neighbors, use their ratings for the item you want to predict a rating for: <https://www.kaggle.com/code/ecemboluk/recommendation-system-with-cf-using-knn>


clustering


MODEL-BASED APPROACHES
Linear model: <https://medium.com/@maxbrenner-ai/matrix-factorization-for-collaborative-filtering-linear-to-non-linear-models-in-python-5cf54363a03c>

Deep Learning: <https://keras.io/examples/structured_data/collaborative_filtering_movielens/> (more explanation: <https://www.kaggle.com/code/jamesloy/deep-learning-based-recommender-systems#Our-model---Neural-Collaborative-Filtering-(NCF)>)

Rating

1.2 M (1M)

+100 000 (80%/20%)


W1

+100 000 
+100 000
+100 000
+100 000
+100 000



----------
### Refs
usar mse
e métricas de retrieval tbm

Continual Collaborative Filtering Through Gradient Alignment


https://medium.com/@maxbrenner-ai/matrix-factorization-for-collaborative-filtering-linear-to-non-linear-models-in-python-5cf54363a03c
https://github.com/maxbrenner-ai/matrix-factorization-for-collaborative-filtering/blob/master/matrix_factorization.ipynb

https://pytorch.org/tutorials/beginner/saving_loading_models.html

https://github.com/noveens/svae_cf/blob/master/main_svae.ipynb





------------
https://induraj2020.medium.com/recommendation-system-using-pyspark-kafka-and-spark-streaming-3eb36b3c3df0

Create a view: https://medium.com/conduktor/getting-started-with-pyspark-kafka-sql-and-ai-e1ac39c8e893

https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.streaming.DataStreamWriter.foreachBatch.html

https://github.com/Siddharth1698/Spotify-Recommendation-System-using-Pyspark-and-Kafka/blob/main/STREAM_WORK-Module.ipynb

https://docs-databricks-com.translate.goog/en/machine-learning/train-model/distributed-training/horovod-spark.html?_x_tr_sl=en&_x_tr_tl=pt&_x_tr_hl=pt-PT&_x_tr_pto=sc


DEPLOY WITH FLASK: https://akbarikevin.medium.com/a-guide-to-deploying-machine-learning-models-with-docker-c259bae7466f
