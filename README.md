# Textual Information Retrieval
Three IR models (Vector Space Model, BM25, Query Likelihood Model) are implemented in single python file main.py

main.py does all the calculations and generates the ranking for the implemented three IR models.

Since different experiments are implemented type of experiment is passed as an argument.
For example: To run experiment 2 execute the script as python main.py 2

Execution time: Each experiment except experiment 4 takes around 50 min to run and generate the output file which
can be then used by trec eval to evaluate the results.

Experiment 4 takes 90 seconds to execute since the number of words are less.

Best results are obtained in experiment 2
