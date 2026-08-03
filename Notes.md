## Day 0
Craeted the project and initialized the repo. The data folder is exterior to the repo since it's very large. The data folder is located at `../data/`. The data folder will be provided seprately, or you can download it from the provided link. The data folder contains the following files:
- 'dev_20240627' the test set for the task.
- 'train' the training set for the task.

The task at hand is the natural langauge to sql translation.

## Day 1
- Created a new notebook `test.ipynb` to test the api calls using gcp, and also to look at the data in the training set.
- The root dir contains two jsons, train, and train_tables, it looks like the train json contains the natural language queries, the db id it needs to work on, and the corresponding sql query. The train_tables json contains the db id, and the corresponding schema of the db. The schema is a list of tables, and each table has a name, and a list of columns, and each column has a name, and a type.
- It contains a total of 9428 examples, and 69 dbs.(This information is from the jsons, not from the data folder)
- train_gold_sql just contains the sql queries from the train set, nothing new.

## Day 2
- building an evaluator.
- Since all the apis have rate limits that are not sufficient for the task, hosting an open source model using quantized weights using vllm and running the queries on it, and after getting the sql queries we run the evaluation locally.

## Day 3
- The evaluator is working, the api calls and evaluation ran all night.
- It achieved a score of 0.66 on the train set with zero shot prompting.
- The prompt included the schema and all the data provided in the train set, and the model was asked to generate the sql query for the given natural language query.
- Check the notebook for the inferences made for the samples that went wrong.
- Moving the evaluation to the dev set now, and will see how it performs on the dev set.
- It has 1534 questions in total, with 11 databases.(This information is from the jsons, not from the data folder)
- Moved from ipynb to scripts, and added the ability to run the evaluation on the dev set, and also added the ability to run the evaluation on a subset of the dev set, by providing a list of ids in a json file.
- The baseline scores are: 
- Zeroshot: 0.61
- Fewshot : 0.63


Discovered that the dev set has some corrupted databases, and the model is not able to generate the correct sql queries for those databases. The corrupted databases are: ['db_0001', 'db_0002', 'db_0003', 'db_0004', 'db_0005', 'db_0006', 'db_0007', 'db_0008', 'db_0009', 'db_0010', 'db_0011']. The corrupted databases are removed from the dev set, and the evaluation is run again on the remaining databases. The new scores are:
- Zeroshot: 0.66
- Fewshot : 0.68

Run the 32 B to see which 