# Continual Learning pipeline for Neural Collaborative Filtering (NCF)
Recommendation systems are widely used in the industry to provide personalised recommendations to users, but research often emphasises the algorithm performance on a static dataset, overlooking the dynamic addition of new users and items.

In this work, we first explore three Neural Collaborative Filtering (NCF) methods for predicting item ratings on the ‘MovieLens 1M’ dataset. Then, using the best model and its optimal configuration, we build an end-to-end engineering solution for continual learning from scratch with Apache Kafka, Apache Spark and Flask.

It starts with an initial offline training of the selected model using 1 million samples from the ‘MovieLens 25M’ dataset. Afterwards, the model is fine-tuned in Spark with the incoming batches of user-item interactions, which are generated from the remaining 24 million samples. Finally, we deploy the model in a Flask server, for inference. In order to ensure horizontal scalability and handle a large number of user-item interactions, the entire pipeline is Docker-ready.

----

**FULL REPORT**: [report.pdf](report.pdf)
