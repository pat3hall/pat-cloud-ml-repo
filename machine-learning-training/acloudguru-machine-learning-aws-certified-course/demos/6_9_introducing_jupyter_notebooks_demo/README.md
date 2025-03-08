------------------------------------------------------
6.9 Introducing Jupyter Notebooks (Amazon SageMaker)
------------------------------------------------------



In this lab, we're going to use a Jupyter Notebook with Amazon SageMaker to use an existing notebook, execute existing code
written by others, and also write and execute our own code. Basic knowledge of Python will be helpful, but not required.



  Note: Downloaded demo files to:
    C:\pat-cloud-ml-repo\machine-learning-training\acloudguru-machine-learning-aws-certified-course\demos\6_9_introducing_juypter_notebooks_demo

The files used in this lab can be found on our GitHub.
   https://github.com/pluralsight-cloud/AWS-Certified-Machine-Learning-Specialty-Labs/tree/main/Introducing-Jupyter-Notebooks-Amazon-SageMaker

Learning objectives
  - Open the existing Jupyter Notebook
  - Execute the Demonstration code
  - Write a new section to the code
  - Perform basic data analytics


    Lab Diagram

                                ---> Markdowns
       SageMaker ---> Notebook
                                ---> Code Cells     --->  Data Objects
                                                    --->  pandas
                                                          External Libraries


Solution
Log in to the Lab Environment

    To avoid issues with the lab, open a new Incognito or Private browser window to log in to the lab. This ensures that your
    personal account credentials, which may be active in your main window, are not used for the lab.  Log in to the AWS Management
    Console using the credentials provided on the lab instructions page. Make sure you're using the us-east-1 region.

Navigate to the Jupyter Notebook

    In the search bar on top, type "Sagemaker" to search for the Sagemaker service.
    Click on the Amazon SageMaker result to go directly to the Sagemaker service.
    Click on the Notebook Instances button to look at the notebook provided by the lab.
    Check to see if the notebook is marked as InService. If so, click on the Open Jupyter link under Actions.
    Click on the AssessmentReviewNotebook.ipnyb file.
    Wait for the kernel to spin up. You'll see a green button that reads Kernel ready in the upper right momentarily when the kernel is finished spinning up.

Demo Orientation

The first section of the lab is intended to demonstrate the basic functionality of Jupyter Notebooks, and provide some familiarization with the basics of Python.

    Run the first code cell to execute basic variable assignment, perform simple arithmetic, and print the result to the console.
    Run the second code cell to do the same, but with multiplication using the earlier assigned variables.
    Run the third code cell to reassign the two previously assigned variables.
    Run the second code cell again to see that the result has changed, since the variables have changed.
    Run the fourth code cell to import the random library, and define the function is_odd_or_even, which we'll use in the next cell.
    Run the fifth code cell to use our earlier function to check whether both our earlier assigned variables, and a random number are either odd or even numbers.

Initializing our Notebook

    Under the Initializing our Notebook section, run the first code cell to import the necessary pandas and matplotlib packages. Both are installed in our environment by default.
    Run the second code cell to import the existing CSV into the dataset variable, and to print how many records (rows) are in the CSV.
    Run the third cell to print out the first several rows in the table.

Basic Data Analysis

    Under the Basic Data Analysis section, run the only code cell to confirm how many unique attempts (rows) exist for each unique question ID.

Diving Deeper

    Under the Diving Deeper section, select the empty code cell, use the dropdown in the toolbar to change it from a Code cell, to a Markdown cell, and add a description about what we're about to do. For example:

        Let's begin by evaluating the success rate of each question and dientify any potential outliers. We can then further examine these outlier questions to ascertain the quality of the question itself, or determine if the original training material needs enhancement.

    Run the Markdown cell to render the text.

    Create a new cell, and enter the following code to perform the necessary analysis:

    success_rate = dataset.groupby('questionId')['isCorrect'].mean()

    plt.bar(success_rate.index, success_rate.values)

    for i, rate in enumerate(success_rate):
        plt.text(i, rate, f'{rate:.0%}', ha='center', va='bottom')

    plt.xlabel('Question ID')
    plt.ylabel('Success Rate')
    plt.title('Question Success Rate')

    plt.xticks(rotation=90)

    plt.show()

    Run the cell to perform the analysis by generating the bar chart graphic.

Conclusion

Congratulations - you just learned how to work with Jupyter Notebooks and create your own notebook!

