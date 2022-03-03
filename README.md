# Textual Information Retrieval
Three IR models (Vector Space Model, BM25, Query Likelihood Model) are implemented in single python file main.py

This project is to be executed on python 3.8

main.py does all the calculations and generates the ranking for the implemented three IR models.

Place this main.py file in the folder containing collections and topics. Haven't pushed the collection and
topics folder in this repository but a screenshot is attached how the folder looks on my machine.

Since different experiments are implemented type of experiment is passed as an argument.
For example: To run experiment 2 execute the script as python main.py 2

Execution time: Each experiment except experiment 4 takes around 45 - 60 min to run and generate the output file which
can be then used by trec eval to evaluate the results.

Experiment 4 takes 2 min to execute since the number of words are less.

Best results are obtained in experiment 2

The index_structures folder contains json files for understanding the data structure used in this project.
The dictionary of dictionaries used in this project is converted to json. It does not have completed data but only few samples.
This is just for understanding the structure and are not used anywhere during the code.
