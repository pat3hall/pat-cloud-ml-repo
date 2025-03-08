--------------------------------------------------
   AWS SageMaker built in Algorithm
--------------------------------------------------
AWS Certified Machine Learning - Specialty (MLS-C01) Exam Guide
Version 2.4 MLS-C01 6 | PAGE

Domain 3: Modeling
  Task Statement 3.1: Frame business problems as ML problems.
  • Determine when to use and when not to use ML.
  • Know the difference between supervised and unsupervised learning.
  • Select from among classification, regression, forecasting, clustering, recommendation, and foundation models.
  
  Task Statement 3.2: Select the appropriate model(s) for a given ML problem.
  • XGBoost, logistic regression, k-means, linear regression, decision trees, random forests, RNN, CNN, ensemble, 
    transfer learning, and large language models (LLMs)
  • Express the intuition behind models.
  
  Task Statement 3.3: Train ML models.
  • Split data between training and validation (for example, cross validation).
  • Understand optimization techniques for ML training (for example, gradient descent, loss functions, convergence).
  • Choose appropriate compute resources (for example GPU or CPU, distributed or non-distributed).
    o Choose appropriate compute platforms (Spark or non-Spark).
  • Update and retrain models.
  o Batch or real-time/online
  
  Task Statement 3.4: Perform hyperparameter optimization.
  • Perform regularization.
  o Dropout
  o L1/L2
  • Perform cross-validation.
  • Initialize models.
  • Understand neural network architecture (layers and nodes), learning rate, and activation functions.
  • Understand tree-based models (number of trees, number of levels).
  • Understand linear models (learning rate).
  
  Task Statement 3.5: Evaluate ML models.
  • Avoid overfitting or underfitting.
    o Detect and handle bias and variance.
  • Evaluate metrics (for example, area under curve [AUC]-receiver operating characteristics [ROC], accuracy, precision, 
    recall, Root Mean Square Error [RMSE], F1 score).
  • Interpret confusion matrices.
  • Perform offline and online model evaluation (A/B testing).
  • Compare models by using metrics (for example, time to train a model, quality of model, engineering costs).
  • Perform cross-validation.
--------------------------------------------------
From: 
  KD nuggets: Understanding Bias-Variance Trade-Off in 3 Minutes
    https://www.kdnuggets.com/2020/09/understanding-bias-variance-trade-off-3-minutes.html

  Bias and Variance in Machine Learning
    https://www.geeksforgeeks.org/bias-vs-variance-in-machine-learning/

Bias vs Variance

  Bias 
    - is the simplifying assumptions made by the model to make the target function easier to approximate.
    - These differences between actual or expected values and the predicted values are known as error or 
       bias error or error due to bias. 
    - Bias is a systematic error that occurs due to wrong assumptions in the machine learning process. 
    - a model with high bias will underfit the data

  Variance 
    - is the amount that the estimate of the target function will change, given different training data. 
    - Variance is the measure of spread in data from its mean position.
    - In machine learning variance is the amount by which the performance of a predictive model changes when 
      it is trained on different subsets of the training data


  Error due to Bias
    - Bias is the distance between the predictions of a model and the true values. 
    - In this type of error, the model pays little attention to training data and oversimplifies the model 
      and doesn't learn the patterns. 
    - The model learns the wrong relations by not taking in account all the features
    - high bias [over simplifying the data] results in unfitting the data

  Errors due to Variance
    - Variability of model prediction for a given data point or a value that tells us the spread of our data. 
    - In this type of error, the model pays an lot of attention in training data, to the point to memorize it 
      instead of learning from it. 
    - A model with a high error of variance is not flexible to generalize on the data which it hasn’t seen before.
    - high variance results in overfitting the data


  Ways to reduce bias:
    Use a more complex model: 
      - One of the main reasons for high bias is the very simplified model
    Increase the number of features: 
      - By adding more features to train the dataset will increase the complexity of the model. 
    Reduce Regularization of the model: 
      - Regularization techniques such as L1 or L2 regularization can help to prevent overfitting and 
        improve the generalization ability of the model. 
      - if the model has a high bias, reducing the strength of regularization or removing it altogether 
        can help to improve its performance.
    Increase the size of the training data: 
      - Increasing the size of the training data can help to reduce bias by providing the model with 
        more examples to learn from the dataset.


  Ways to reduce variance:
    Cross-validation: 
      - By splitting the data into training and testing sets multiple times, cross-validation can help identify 
        if a model is overfitting or underfitting and can be used to tune hyperparameters to reduce variance.
    Feature selection: 
      - By choosing the only relevant feature will decrease the model’s complexity. and it can reduce the variance error.
    Regularization: 
      - We can use L1 or L2 regularization to reduce variance in machine learning models
    Ensemble methods: 
      - It will combine multiple models to improve generalization performance. 
      - Bagging, boosting, and stacking are common ensemble methods that can help reduce variance and improve 
        generalization performance.
    Simplifying the model: 
      - Reducing the complexity of the model, such as decreasing the number of parameters or layers in a 
        neural network, can also help reduce variance and improve generalization performance.
    Early stopping: 
      - Early stopping is a technique used to prevent overfitting by stopping the training of the deep learning 
        model when the performance on the validation set stops improving.

--------------------------------------------------
Decision and Classification Trees, Clearly Explained!!!
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=_L39rN6gz7Y

  Decision Tree
    - makes a statement, then makes a decision based on whether the statement was true or false
    Classification Tree
      - when a decision tree classifies things into categories, it called a Classification tree
    Regression Tree
      - when a decision tree predicts numeric values, it called a Regression tree

  Classification trees
    - can mix decision data types in the same tree (True/False statements, value statements)
    - numeric threshold can be different for the same data ( value > 20, value < 30)
    - final classification can be repeated
    - Note: true statements are assumed to be to the left, and false statements to the right
    
  Classification trees terminology
    root node or root: 
      - top of tree
    internal nodes or branches
      - have arrows above pointing to them and arrows below point away from them
    leaves nodes
      - have arrows above pointing to them BUT NO arrows below point away from them (bottom of trees)

   Impure node
     - contain a mixture results for item to be predicted 

     How to quantify the impurity of a prediction leaf:
       - methods to quantify leaf impurity include:
         Gini Impurity
         Entropy
         Information Gain
       - methods to numerically quantify the impurity of a leaf are similar, so focus is on Gini Impurity
         which is the most straightforward

      Step 1. calculate the gini impurity value for each potential branch

       Gini Impurity for boolean branch:
         
         Gini Impurity for a leaf = 1 - (probability of 'True')**2 - (the probability of 'False')**2

         Total Gini Impurity      = weighted average of the Gini Impurities of the leaves
    
          Example:
                               love popcorn
               /-----------------|   |-------------------\
               V     True               False            V
        love Cool as Ice                              love Cool as Ice
        Yes      No                                   Yes       No
        1         3                                    2         1

        Leaf Gini Impurity                          Leaf Gini Impurity 
        = 1 - (1/(3 +1))**2 - (3 / (3 + 1))**2               = 1 - (2/(2 +1))**2 - (1 / (2 + 1))**2 
        = 0.375                                              = 0.444

         Weighted average                            Weighted average
         = [4 / ( 4 + 3)] * 0.375 = 0.214              = [3 / ( 4 + 3)] * 0.444 = 0.190

         Total Gini Impurity = 0.214 + 0.190 = 0.404


     Gini Impurity calculation for numeric values branches:
       - sort samples from lowest to highest value for the variable (e.g. age)
       - calculate the average value between samples (ordered sample 1 age: 7, sample 2: 12, average: 9.5)
       - calculate the Gini Impurity values for each average age
       - use average weight with lowest gini impurity score


                               age < 9.5   
               /-----------------|   |-------------------\
               V     True               False            V
        love Cool as Ice                              love Cool as Ice
        Yes      No                                   Yes       No
        0         1                                    3         3

        Leaf Gini Impurity                          Leaf Gini Impurity 
        = 1 - (0/(0 +1))**2 - (1 / (0 + 1))**2               = 1 - (3/(2 +3))**2 - (3 / (3 + 3))**2 
        = 0                                                  = 0.5

         Weighted average                            Weighted average
         = [1 / ( 1 + 6)] * 0  = 0                     = [6 / ( 1 + 6)] * 0.5  =  0.429

         Total Gini Impurity = 0 + 0.429 = 0.429


         Calculate the Gini Impurity value for each average weight
            Average Weight     Gini Impurity
               9.5                0.429
              15                  0.343    --------> tied for lowest impurity value *
              26.5                0.476
              36.5                0.476
              44                  0.343    --------> tied for lowest impurity value *
              66.5                0.429

        * since tied value, you can choice either, and choice age < 15`

    step 2: select branch with lowest gini impurity value
       - lowest gini impurity value equates to most pure branch available

      Use tree branch with lowest impurity:

        Gini Impurity for Loves Popcorn = 0.404
        Gini Impurity for Loves Soda    = 0.214   ---> has lowest value, so it leaves has lowest impurity **
        Gini Impurity for Age < 15      = 0.343

     Step 3: Repeat process for each now leaf branch
          

  Decision Tree
    - calcuate the Gini impurity value for each decision node,  and use the decision node with the
      lowest Gini impurity value
     - stop at pure nodes wich are called leaves

    Leaf Output
      - after determine all leaf, need to assign output values for each leaf
      - leaf output is the category that has the most votes


     Handling leafs with too few results
        - hard to confidence in a leaf that has too few results (could cause overfitting)
        methods to deal with overfitting
          Pruning
          limits on how trees grow
            - require minimum of results per leaf
            - this may results with impure leafs - this is handled by calculate probability with
              output value based on the majority

         Cross Validation
           - use cross validation to determine which decision tree works best

--------------------------------------------------
StatQuest: Random Forests Part 1 - Building, Using and Evaluating
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=J4Wdy0Wc_xQ

  Summary:
    - Random Forests make a simple, yet effective, machine learning method. 
    - They are made out of decision trees, but don't have the same problems with accuracy. 
    - In this video, I walk you through the steps to build, use and evaluate a random forest.


  Decision Trees
    - easy to build and easy to use, 
    - trees have one aspect that prevents them from being the ideal too for predictive learning,
      namely accuracy
    - they work greate with the data used to create them, but are not flexible when it comes to classifying
      new samples

  Random Forest
    - combines the simplicity of decision trees with the flexibility result in a vast improvement in accuracy

  Random Forest flow
   1. Create a Bootstrap dataset
     - to create a bootstrapped dataset that is the same size as the original, we just randomly select sameples
       from the original dataset
     - allow to pick the same sample more than once

   2. Only considering a random subset of variables at each step
     - Build the decision tree as usual, BUT only considering a random subset of variables [columns] at each step

   3. Go back to Step and repeat
     - make a new bootstrapped dataset and build a tree considering a random subset of variables at each step
     - ideally, you do this 100's of times

   Outcome:
     - using a bootstrapped sample and considering only a subset of the variables at each step results in a
       wide variety of trees
     - the variety is what makes random forests more effective that individual decision trees

    How to use to predict:
      - for each new sample, run the  sample down all the generated trees, and sum the results [votes] 
        from the trees
        - the result with the votes wins

   Bagging
     - bootstapping the data plus using the aggregate to make a decision is called "bagging"

   Out-of-Bag-dataset
     - typically, about 1/3 of the original data does not end up the bootstrapped data
       - think of it as out-of-boot-dataset
     - use out-of-bag-dataset to 

   Measure Random Forest accuracy
     - can measure how accurate our random forest by the proportion of Out-of-bag samples which
       are correctly classified by the random forest
     Out-of-Bag-error
       - the proportion of Out-Of-Bag samles thate were incorrectly classified

   Variables Per Step
     - compare the Out-of-Bag error for a random forest buit using x (e.g. 2) variables per step against 
       using y (e.g. 3) variables per step
        - choose the most accurate random forest


   Full Random Forest flow
   1. Build a Random Forest
   2  Estimate the accuracy of a Random Forest
   3. Change the number of random variables per step and repeat steps 1 and 2
   4. Repeat many times, then choose the most Random Forest accurate model
   Note:
     - Typically, we start by using the square of the number of variables, and then try a few 
       settings above and below that value
--------------------------------------------------
StatQuest: Random Forests Part 2: Missing data and clustering
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=sQ870aTKqiM

  Summary:
    - NOTE: This StatQuest is the updated version of the original Random Forests Part 2 and includes two minor corrections.
    - Last time we talked about how to create, use and evaluate random forests. 
    - Now it's time to see how they can deal with missing data and how they can be used to cluster samples, even 
      when the data comes from all kinds of crazy sources.

  Missing Data types
    - missing data in the original dataset used to create the random forest
    - missing data in a new sample that we want to categorize
    
   Handling Missing data in the original dataset
     - make an initial guess that could be bad, then gradually refine the guess until it is (hopefully)
       a good guess
     - initial categoric/boolean value is the most common variable value for the target value of the sample 
     - initial numeric value is the median value for the target value of the sample

   Refining the initial values/guesses
     - first, determine which samples are similar to the one with missing data
     step 1: Build a Random forest
     step 2: run all the data down all of the trees
     similarity:
       - sample data that ends up at the same leaf (terminal) node
       - keep track of similar samples using a "Proximity Matrix"
       - the proximity (or a similarity) between two examples is a number indicating how "close" those two 
         examples are
     step 3: Calculate proximity matrix  
         Proximity Matrix
           - has a row for each sample
           - has a column for each sample
           - if a sample represented in a column ends up in the same leaf node as a sample 
             represented in row, a 1 [add 1] at this row/column location
           - repeat - run all the data down all the the decision trees

             Fill-in Missing Values

         Chest   Good    Blocked  Weight   Heart 
         Pain    Blood   Arteries          Disease
                 Circ
        -------------------------------------------
          No      No        No       125      No
          Yes     Yes       Yes      180      Yes
          Yes     Yes       No       210      No
          Yes     Yes       ???      ???      No
        -------------------------------------------

      step 3a: Calculate proximity matrix
          | 1  2  3  4
        --|------------
         1|    2  1  1         
         2| 2     1  1
         3| 1  1     8
         4| 1  1  8  
       
      step 3b: Calculate weighted proximity matrix
           - divide each proximity value by the total number of trees (e.g. 10 trees)
          |  1   2    3    4
        --|------------------
         1|     0.2  0.1  0.1         
         2| 0.2      0.1  0.1
         3| 0.1 0.1       0.8
         4| 0.1 0.1  0.8  
       
       step 4: Fill-in missing values
          for yes/no: calculate frequency of "yes" and frequency of "no"
             - if 1 "yes", 2 "no", and 1 missing in the 4 samples for Block Arteries
               yes = 1/3
               no = 2/3 

            weight frequency of yes = freqency of yes  x  weighted for "yes"

            weight for "Yes" =   Proximity of "yes"  /   All proximities


             Block Arteries Yes occurred only in sample to with "0.1" proximity value

             for sample 4 (has missing value), the proximity values are 0.1 + 0.1 + 0.8

             sample 4 weight for yes = 0.1 / (0.1 + 0.1 + 0.8) = 0.1 
                  -> since only sample 2 has 'yes'

             sample 4 weight for no  = (0.1  + 0.8) / (0.1 + 0.1 + 0.8)  = 0.9
                         -> since Sample 1 and 3 have no


             weighted frequency for "yes" = 1/3  x   0.1   = 0.03

             weighted frequency for "no" =  2/3  x   0.9   = 0.6

            
            Since 'No' weighted freqency is Much higher than "yes" weighted frequency,
               the refined value (improved guess) to use for the missing Block Arterties is "No"

              
               
          for sample 4 missing Weight value: 
                - calculated a weighted average for each sample, and sum the results

           sample's weight average = sample weight   x   Sample weighted average weight 

             sample 1 weight average = 125  x  (0.1  / [0.1 + 0.1 + 0.8]) = 125  x 0.1 = 12.5

             sample 2 weight average = 180  x  (0.1  / [0.1 + 0.1 + 0.8]) = 180  x 0.1 = 18.0

             sample 3 weight average = 210  x  (0.8  / [0.1 + 0.1 + 0.8]) = 210  x 0.8 = 168.0

             sum of sample weight average = 12.5 + 18.0 + 168.0 = 198.5

             Sample 4 revise value (improved value) based on proximity weighted average is: 198.5



       step: repeat steps 1 to 4 6 or 7 times, until the missing values converge 
            (missing values no longer change when recalcuated)
            - That is, build a random forest, run the data through the trees, recalcuate the
              proximities and recalculate the missing values


       Proximity of "1" in a proximity matrix
         - means the samples are as close as can be
         - means: 1 -  proximity value = distance
           - therefore, proximity of '1' means as close as can be
                        proximity of '0' means as far away

            - thus, proximity  map can be used to draw a heat map and MDS plot 
              to show how the samples are related to each other


       
     MDS (Multidimensional scaling) plot 
       - a means of visualizing the level of similarity of individual cases of a data set. 
       - MDS is used to translate distances between each pair of 'n' objects in a set into a configuration 
         of 'n' points mapped into an abstract Cartesian space.


    Handling missing data in a new sample that we want to categorize

       Imagine we had already built a Random Forest with existing data and wanted to classify
       this new patient (to determine if they have heart disease)

         Chest   Good    Blocked  Weight   Heart 
         Pain    Blood   Arteries          Disease
                 Circ
        -------------------------------------------
          Yes     No        ???      168        

       step a:
         - create two copies (samples) of the data, one with "yes" for "heart disease" and
           one with "no" for "heart disease"
       step b:
         - use the iterative method previously discussed to make a good guess about the missing value
           - use initial guess values
           - run the 2 samples down the trees in the forest
              - example for "yes" heart disease sample with initial "yes" Blocked Arteries, 
                  - this option was correctly lableled "yes" in all 3 trees
              - example for "no" heart disease sample with initial "no" Blocked Arteries, 
                  - this option was correctly lableled "no" in only 1 tree
         
              - therefore, use the "yes" for heart disease with "yes" for block arterties

--------------------------------------------------

     Proximity:
       https://stats.stackexchange.com/questions/137358/what-is-meant-by-proximity-in-random-forests
       - The term "proximity" means the "closeness" or "nearness" between pairs of cases.

       - Proximities are calculated for each pair of cases/observations/sample points. 
       - If two cases occupy the same terminal node through one tree, their proximity is increased by one. 
       - At the end of the run of all trees, the proximities are normalized by dividing by the number of trees. 
       - Proximities are used in replacing missing data, locating outliers, and producing illuminating 
         low-dimensional views of the data. 
--------------------------------------------------
AdaBoost, Clearly Explained
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=LsK-xG1cLYA

  Summary:
    - AdaBoost is one of those machine learning methods that seems so much more confusing than it really is. 
    - It's really just a simple twist on decision trees and random forests.


  Note: This StatQuest show how to combine AdaBoost with Decision Trees, because that is the most
        common way to use AdaBoost


   Adaboost 3 main concepts:
      Stump
        - a tree with just one node and two leaves is called a stump
        - a stump can use only 1 variable to make a decision
        - stumps are NOT great at making accurate classification
        - stumps are technically "weak learners"
       1. AdaBoost stump 
         - Adaboost is forest of stumps rather than trees
       Votes:
         - in random forest, each tree has a 1 vote on the final classification
       2. Adaboots votes
         - some stumps get more say in the final classification than others
       tree independents
         - in a Random Forest, each decision tree is made independently of the others
         - that is, decision tree order does not matter
       3. Adaboost Order
         - a forest of stumps made with Adaboost, order is important
         - the errors the first stump makes, influence how the 2nd stump is made, 
           the errors the 2nd stump makes, influence how the 3rd stump is made, etc.


   Adaboost 3 main concepts review:
     1. Adaboost combines a lot of "weak learners" to make classifications. 
        The weak learners are almost always stumps

     2. some stumps get more say in the classification than others

     3. Each stump is made by taking the previous stumps errors in to account

   How to create a forest of stumps using AdaBoost
     step 1:
       - give each sample a weight that indicates how important it is to be correctly classified
       - at start, each sample gets the same weight: 1 / (toral number of samples)
       - after we make the first stump, the weights will change in order to guide how the next
         stump is created
      step 2: classify each sample
        - for each boolean variables, classify each sample, e.g.
                 

                               chest pain  
               /-----------------|   |-------------------\
               V Yes Heart Disease    No  Heart Disease  V
        correct  incorrect                           correct  incorrect
          3         2                                    2         1

        - for each numeric variables, classify each sample (as in decision tree video)
           - sort samples from lowest to highest value for the variable (e.g. age)
           - calculate the average value between samples (ordered sample 1 age: 7, sample 2: 12, average: 9.5)
           - calculate the Gini Impurity values for each average age
           - use average weight with lowest gini impurity score

      step 3: calculate gini index for each stump

                               chest pain  
               /-----------------|   |-------------------\
               V Yes Heart Disease    No  Heart Disease  V
        correct  incorrect                           correct  incorrect
          3         2                                    2         1          --> Gini index = 0.47
                                                                              --> Total error = (1 + 2) / 8 = 3/8

                               block Arteries
               /-----------------|   |-------------------\
               V Yes Heart Disease    No  Heart Disease  V
        correct  incorrect                           correct  incorrect
          3         3                                    1         1          --> Gini index = 0.5
                                                                              --> Total error = (1 + 3) / 8 = 0.5

                               weight > 176
               /-----------------|   |-------------------\
               V Yes Heart Disease    No  Heart Disease  V
        correct  incorrect                           correct  incorrect
          3         0                                    4         1          --> Gini index = 0.2
                                                                              --> Total error = 1/8


        step 4: Use the stump with the lowest Gini index
             -> use "weight > 176" stump as first stump in the forest

        step 5: Determine how much say the stump will have in the final classification
                - determined by how well it classifies the samples
                - total error for a stump is the sum of the weights associated with the incorrectly 
                  classified samples


            Sample weights and total errors
              - Because all of the sample weights add up to 1, the "total error" will always be 
                between 0 for a perfect stump, and 1 for a horrible stump
            Total Errors
              - Total error is used to determine "amount of say" a stump has in the final classification
                with the following formula:

                Amount of Say = (1/2) log ( [1 - totalError] / totalError)

                   Note: If totalError is 1 or 0, then this equation will freak out.
                         In practice, a small error term is added to prevent this from happening

             Amount of Say:
               - when the total error is small, then the amount of say is relatively large positive value
               - when a stump is no better at classification that flipping a coin and total error = 0.5,
                 then the amount of say will be 0
               - when the total error is close to 0 meaning the stump gives you to opposite classification,
                 then the amount of say is relatively large negative value


                 Chest   Blocked  Weight   Heart      Sample
                 Pain    Arteries          Disease    Weight
                ------------------------------------  -------
                  Yes     Yes      205      Yes         1/8
                  No      Yes      180      Yes         1/8
                  Yes     No       210      Yes         1/8
                  Yes     Yes      167      Yes         1/8   --> incorrectly classified by "weight > 176"  No branch 
                                                                  Therefore, sample weight needs to be adjusted 
                  No      Yes      156      No          1/8
                  No      Yes      125      No          1/8
                  Yes     No       168      No          1/8
                  Yes     Yes      172      No          1/8

            With Patient Weight > 176
                the total errror is 1/8
                Amount of Say = (1/2) log ( [1 - 1/8] / 1/8) = 1/2 * Log (7)  = 0.97

               

        step 6: Adjust sample weights

             - Note: initial sample weights were all set to:  1 / (number of samples)

             Incorrectly Classifed New Sample Weight:
               - formula is used to increase the "Sample Weight" for the incorrecly classified sample(s)
               - when the "amount of say" is relatively large (i.e. the last stump did a good job
                 classifying samples), then we will scale the previous "sample weight" with a relatively large number 
                 That is, the new sample weight will be much larger than previous one
               - when the "amount of say" is relatively large (i.e. the last stump did NOT do a good job
                 classifying samples), then we will scale the previous "sample weight" with a relatively small number.
                 That is, the new sample weight will be only a little larger than the previous one

               Incorrectly classied new sample weight formula:

                  New Sample   =   Sample weight  x  e**(amount of say)
                    weight 

                               =   (1/8)  x     e**(0.97) = (1/8)  x 2.64 = 0.33    --> for sample 4
     
                                Note: 1/8 = 0.125


             Correctly classied new sample weight:
               - formula is used to increase the "Sample Weight" for the correcly classified sample(s)
               - when the "amount of say" is relatively large (i.e. the last stump did a good job
                 classifying samples), then we will scale the previous "sample weight" with a value close to 0
                 That is, the new sample weight very small
               - when the "amount of say" is relatively small (i.e. the last stump did NOT do a good job
                 classifying samples), then we will scale the previous "sample weight" by a value  close to 1
                 That is, the new sample weight will be just a little smaller than the previous one
               - 

             Correctly classied new sample weight formula:


                New Sample   =   Sample weight  x  e**-(amount of say)
                  weight 

                             =   (1/8)  x     e**(-0.97) = (1/8)  x 0.38 = 0.05    --> for samples 1 - 3, 5 - 7
     
                                Note: 1/8 = 0.125

           Normalize New Sample weights
              - sum of new sample weights need to equal one, so new values will need to be normalized
              - to normalize:

                        Normalized      = new Sample weight  / (sum of all new sample weights)
                        sameple weight


                                                                initial      Normalized  Incremental Sum
                 Chest   Blocked  Weight   Heart      Sample   New Sample    New Sample  of new sample 
                 Pain    Arteries          Disease    Weight    Weight       Weights     weights
                ------------------------------------  -------  ----------    ----------  -------------
                  Yes     Yes      205      Yes         1/8       0.05          0.07       0.07
                  No      Yes      180      Yes         1/8       0.05          0.07       0.14
                  Yes     No       210      Yes         1/8       0.05          0.07       0.21
                  Yes     Yes      167      Yes         1/8       0.33          0.49       0.70
                                                              
                  No      Yes      156      No          1/8       0.05          0.07       0.77
                  No      Yes      125      No          1/8       0.05          0.07       0.84
                  Yes     No       168      No          1/8       0.05          0.07       0.91
                  Yes     Yes      172      No          1/8       0.05          0.07       0.98
                                                                 ------        -------
                                                             Sum: 0.68       sum: ~1 (0.98)


        step 7: Use new [normalized] sample weights to make the next stump in the forest 
           weight gini index approach
             - in theory, we could use the "Sample Weight" to calculate the "Weight Gini Index" to
               determine which variable should split the next stump
             - weight Gini index would put more emphasis on correctly classifying sample 4 (previously
               misclassified) since this sample has the largest "Sample Weight"  (0.49)
           new collection of samples approach
             - can make a new collection of samples that contain dumplicate copies of the sample with
               the largest "sample weight"
             - start by making a new dataset that is the same size as the original (e.g. 8 samples)
             - pick a random number between 0 and 1
                 - see where the random number falls in the "Incremental Sum of new sample weights"
                   (note: using sample weights like a distribution)
                   - if between 0.00 and 0.07, the put sample 1 in the new collection
                   - if between 0.07 and 0.14, the put sample 2 in the new collection
                   - if between 0.14 and 0.21, the put sample 3 in the new collection
                   - if between 0.21 and 0.49, the put sample 4 in the new collection
                     - etc.
              - add samples based on the selected random numbers until new collection is the same
                size as the original collection
              - give all the samples equal sample weights 
              - however, the next will likely need to correctly classify the original sample 4
                since it would typically be duplicated 4 times

        step 8: repeat steps 1 to 7 on the new collection
           - find the stump that does the best job classifying the new collections of samples
           - this results in the errors the first tree makes influence how the 2nd tree is made, 
             and the errors that the 2nd tree makes influence how the 3rd tree is made, etc.


       How a forest of stumps created by AdaBoost makes classifications
          - calculate the "amount say" for each stump
          - sum "amount of say" for each classification (Has heart disease  and   does not have heart disease)
            - classification with largest "amount of say" is selected

         Has Heart Disease    Amount of Say                 Amount of Say     Does Not have
                                                                               Heart Disease 
            stump 1     --->    0.97                            0.41  <-----      stump 5

            stump 2     --->    0.32                            0.81  <-----      stump 6

            stump 3     --->    0.78                            1.23   : total

            stump 4     --->    0.63

                           total: 2.7    

               
  Review
   The 3 ideas behind AdaBoots are:
     1. Adaboost combines a lot of "weak learners" to make classifications. 
        The weak learners are almost always stumps

     2. some stumps get more say in the classification than others

     3. Each stump is made by taking the previous stumps errors [mistakes] into account
        - if we have a "Weighted Gini Function", then use it with the "sample weights"
        - otherwise, use the "Sample weights" to make a new dataset that reflects those weights


--------------------------------------------------
 From: O'Reilly Hands-ON Machine Learning with Scikit-Learn, Keras, and TensorFlow


6. If your AdaBoost ensemble underfits the training data, which hyperparameters should you tweak, and how?

  -> AdaBoost ensemble increases the weight of the 'unfitting instances' (misclassified) after each estimator stage, so 
  increasing the number of stages/estimators ('n_estimators' hyperparater) can be used to reduced underfitting. Reducing 
  the regularization of the estimators used, can also be used to reduce underfitting.

  page 222 - 223: One way for a new predictor to correct it predecessor is to pay a bit more attention to the training instances 
  that the predecessor 'underfit'. This results in new predictors focusing more and more on the hard cases. This is the 
  technique used by 'AdaBoost'.

  For example, when training an 'AdaBoost classifer', the algorithm first trains a base classifier (such as 'decision tree') 
  and uses it to make predictions on the training set. The algorithm then increases the relative weight of the misclassified 
  training instances. Then it trains a second classifier, using the updated weights, and againg makes predictions on the 
  training set, update the instance weights, and so on (see Figure 7-7 [AdaBoost sequential training with instanc weight updates])

  page 226: Another popular boosting algorithm is 'gradient boosting' (...). Just like AdaBoost, gradient boosting works by 
  sequentially adding predictors to an ensemble, each one correcting it predecessor. However, instead of tweaking the instance 
  weights at every interation like AdaBoost does, this method tring to fit the new predictor to the 'residual errors' make 
  by the predictor.

  page 226: If your AdaBoost ensemble is overfitting the training set, you can try reducing the number of estimators or more 
  strongly regularizing the base estimator

  book answer: If your AdaBoost ensemble underfits the training data, you can try increasing the number of estimators or 
  reducing the regularization hyperparameters of the base estimator. You may also try slightly increasing the learning rate.

--------------------------------------------------
Gradient Boost Part 1 (of 4): Regression Main Ideas
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=3CC4N4z3GJc&t=87s


  Summary:
    - Gradient Boost is one of the most popular Machine Learning algorithms in use. 
    - This video is the first part in a series that walks through it one step at a time. 
    - This video focuses on the main ideas behind using Gradient Boost to predict a continuous value, 
      like someone's weight. We call this, "using Gradient Boost for Regression". 
    - In the next video, we'll work through the math to prove that Gradient Boost for Regression really is 
      this simple. 
    - In part 3, we'll walk though how Gradient Boost classifies samples into two different categories, 
      and in part 4, we'll go through the math again, this time focusing on classification.
  

  Gradient Boost for Regresssion
    - when gradient boost is used to predict a continous value, like weight, it is being used
      for regression
    - gradient boost algorithm looks complicated because it was designed to be configured in a wide
      variety of ways, but 99% of the time only one configuration is used to predict continous values,
      and one configuration is used to classify samples into different categories

  Gradient Boost vs AdaBoost
    AdaBoost
      - AdaBoost starts by building a very short tree, called a 'stump', from the training data, and the
        amount of say that the stump has on the final output is based on how well it compenstated for those
        previous errors
      - Then AdaBoost builds the next stump based on errors that the previous stump made
      - AdaBoost continues to make stumps in this fashion until it has made the number of stumps you 
        requested, or it has perfect fit
    Gradient Boost
      - starts by make a single leaf, instead of a tree or stump
      - this leaf represents an initial guess for the weights of all of the samples
      - when trying to predict a continuous value like 'weight', the first guess is the average value
      - then gradient boost builds a tree - (like AdaBoost) this tree is based on the errors made
        by the previous tree
      - the gradient Boost tree is usually larger than a stump, but it restricts the size of the tree
      - gradient boost scales all tree by the same amount
      - in practice, the maximum number of leaves is often set between 8 and 32
      - gradient boost builds another tree based on the errors made by the previous tree, and it scales the tree
      - it continuous to build trees in this fashion until it has made the number of trees you asked for,
        or additional trees fail to improve the fit
     
       Note: The term "Puedo Residual" is based on "Linear Regression", where the difference
             between "Observed" values and the "Predicated" values results in "Residuals"

     Using Gradient Boost to predict weight

       Weight Dataset:
                               
          height   Favorite    Gender   Weight   Residual1  predicted  residual2  predicted  residual3
           (m)      color                (kg)    (error)    weight 1    (error)   weight 2
        ---------------------------------------- -------    --------    -------  ----------  ---------
           1.6       Blue        Male     88       16.8       72.9       15.1      74.4        13.6
           1.6       Green      Female    76        4.8       71.7        4.3      72.1         3.9
           1.5       Blue       Female    56      -15.2       69.7      -13.7      68.4       -12.4
           1.8       Red         Male     73        1.8       71.6        1.4      71.9         1.1
           1.5       Green       Male     77        5.8       71.6        5.4      71.9         5.1
           1.4       Blue       Female    57       -14.2       69.7      -12.7      68.4       -11.4
                                     ave:71.2

        1. Start with average weight as residual, build 1 tree, calculate tree's residuals
           - Start with Average Weight: 71.2 as the predicted weight for all samples
           - build 1st tree restricted to 4 leaves
             - by restricting the total number of leaves, we get fewer leaves than residuals 
             - replace leaves with multiple residuals with their average value e.g.: ([1.8 + 5.8]/ 2) = 3.8
           - calculate new predicted weight for tree1: 
                 average weight + learning rate x residual (e.g. 71.2 + 0.1 x 16.8 = 72.9)

               tree 1:
                               Gender=F
                             T/        \F
                      Height<1.6     Color NOT blue
                      /     \            /       \
              -14.2,-15.2   4.8       1.8,5.8     16.8
               replace with averages:
               -14.7        4.8         3.8       16.8
               predicted weight without learning rate (or set to 1)
               56.5         76          75        88.0
               predicted weight with 0.1 learning rate 
               69.7         71.7        71.6      72.9
 
           low Bias and high variance:
              - model fits the training data to well (observed weight = predicted weight = 88)

         Learning Rate:
           - deals with variance by using a "learning rate" to scale the contribution from the new tree
           - a value between 0 and 1
           - scaling the tree by the "learning rate" results in a small step in the right direction
           - taking lots of small steps in the right direction results in better predictions with
             a testing dataset (i.e. lower variance)

        2. build 2 tree and calculate tree's predicted weight, and new residuals
           - build a tree to predicted the psuedo residual from the previous tree
              - error (previous psuedo residual) : Observied weight - predicted weight
           - calculate new predicted weight for tree2: 
                 average weight + learning rate x residual1  + learning rate x residual2 
                 (e.g. 71.2 + 0.1 x 16.8 + 0.1 x 15.1 = 74.4 

           - note: tree 2 has same branches as tree1, but in practice, branches would vary

               tree 2:
                               Gender=F
                             T/        \F
                      Height<1.6     Color NOT blue
                      /     \            /       \
              -12.7,-13.7   4.3       1.4,5.4     15.1
               replace with averages:
               -13.2        4.3         3.4       15.1
               predicted weight with 0.1 learning rate 
               68.4         72.1        71.9      74.4

        3. repeat - build next tree and calculate tree's predicted weight, and new residuals

        4. stop when: 
           - keep making trees until we reach the maximum specified number of trees OR
             adding additional trees does not significanlty reduce the size of the residuals

        5. Use test data to predict weights
            run test data through all the trees and calcuate the predicted weight:
            LR: learning rate
            RES: residucal value
              ave weight + LR * RES1 + LR * RES2 + LR * RES3 + ....
              ave weight + LR * SUM(RES1 + .. + RESN)

  Gradient Boost for Regression summary:
    - start with a leaf that is the average value of the variable we want to predict
    - add a tree based on the "Residuals", the difference between the "Observed" values and the
      - scal the  tree's contribution to the final prediction with a "Learning Rate" (i.e. 0.1)
    - add another tree based on the new "Residuals"
    - keep adding tree based on the errors made by the previous tree (that is, the new "Residuals")
      "Predicted values"

--------------------------------------------------

Gradient Boost Part 2 (of 4): Regression Details
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=2xudPOBz-vs

  Summary:
    - This video is the second part in a series that walks through it one step at a time. 
    - This video focuses on the original Gradient Boost algorithm used to predict a continuous value, 
      like someone's weight. We call this, "using Gradient Boost for Regression"
               
       Weight Dataset:
                               
          height   Favorite    Gender   Weight   
           (m)      color                (kg)    
        ---------------------------------------- 
           1.6       Blue        Male     88     
           1.6       Green      Female    76    
           1.5       Blue       Female    56   


     Input Data {(x_i,y_i)} i=1 to n
         x_i: each sample row (e.g. height, color, Gender values in a row)
         y_i: each target (e.g weight)
         n: number of rows in dataset

     Differentiable Loss Function:  L(y_i,F(x))
       - used to evaluates how well we are predicting the target
       - F(x) function gives you the predicted values

       Gradient Boost for Regression Most propular Loss function:
         1/2 (observed - predicted)**2

         residual (error) = 1/2 (observed - predicted)

         Therefore, loss function = residual**2

           Note: Linear regression Loss function uses (no 1/2): (observed - predicted)**2

       Loss Function include 1/2 because it simplifies the differential loss function
        
           (d / d predicted )  (1/2 [observed - predicted]**2
                  =  (2/2 [observed - predicted] x -1      # Note: multiple by the derivative "-Predicted" which is -1
                  =  -(observed - predicted)               # left with the negative residual 




    Step 1: Initial model with a constant value: 

              F_0(x) = argmin_γ ∑ L(y_i, gamma)  where ∑ is i=1 to n
                          γ
            - start by initializing the model with a constant value
              L(y_i, γ) -> loss function -> 1/2(observed - predicted)**2
                    y_i: observed values
                    γ: predicted values
                    argmin_γ (over γ) -> means we need to find a "predicted" value the minimums this sum

                    Note: We could  use gradient descent to find the optimal value for predicted,
                          but we can also just solve for it:
                         
                    loss function = 1/2(88 - predicted)**2 + 1/2(76 - predicted)**2 + 1/2(56 - predicted)**2

                    take derivation with respect to predicted:  d/ d predicted
                                    -(88 - predicted) + -(76 - predicted) + -(56 - predicted)
                    set the sum of the derivates to '0'
                                 -(88 - predicted) + -(76 - predicted) + -(56 - predicted) = 0
                                 predicted = (88 + 76 + 56)/3  
                                 predicted = average of the observed weights

                     
                    argmin_γ is γ that minimizes the sum is the average weight

                initial predicted value:
                   F_0(x) = argmin_γ ∑ L(y_i, gamma)  where ∑ is i=1 to n
                   F_0(x) =  (88 + 76 + 56)/3  = 73.3

               Finished step 1: initial the model with a constant FR_0(x) = 73.3
                   in other words, we created a leaf that predicts all samples will weigh 73.3


    Step 2: For m = 1 to M:
        - we will make M trees, but in practices, most people set M=100 and make 100 trees
        - 'm' refers to an individual tree. So when 'm = 1', this is the first tree

       A. Compute r_i,m = -[ ∂L(y_i,F(x_i)) / ∂F(x_i) ] for F(x)=F_m-1(x)  for i = 1,...,n

           ∂L(y_i,F(x_i)) / ∂F(x_i) 
             - just the derivative of the "loss function' with respect to the predicted value 
             - this derivate is the Gradient that Gradient Boost is named after
             which is:
                  (d / d predicted )  (1/2 [observed - predicted]**2 = -(observed - predicted)
             therefore:
                  -[ ∂L(y_i,F(x_i)) / ∂F(x_i) ] = (observed - Predicted) = residual
             Now, we plug F_m-1(x) in predicted
                       (observed -  F_m-1(x))
                       since F_0(x) is the leaf set to 73.3, we plug in 73.3
                       (observed -  73.3)

             compute r_i,m
                 'r' is short for 'residual', i is the sample number, and m is the tree that were trying to build
                 - r_i,m values are technically called psuedo residuals

                     r_1,1  = 88  - 73.3 =  14.7
                     r_2,1  = 76  - 73.3 =   2.7
                     r_3,1  = 56  - 73.3 = -17.3

       b. Fit a regression tree to r_im values and create terminal regions R_j,m for j = 1 ... J_m

            - build a regression tree to fit/predict the residuals instead of the weights
            - use height, favorite color, and gender to predict the residual

                      Height<1.55      Note: Just a stump and Gradient boost always uses larger trees
                      /        \    
                  -17.3       14.7,2.7
                  R_1,1         R_2,1       Note: It does not matter which leaf gets which terminal region label,
                                                  but once a leaf is labeled, we need to keep track of it

              create terminal regions   R_jm for j = 1 ... J_m 
                - leaves are the terminal regions R_j,m  where m is the index for the tree '
                   and j is the index for each leaf the tree
                - since tree has 2 leaves, J_m = 2
                - assigned terminal regions labels R_1,1 and R_2,1

       c. For j = 1...J_m compute  γ_j,m = argmin_γ ∑ L(y_i,F_m-1(x_i) + γ)   for x_i element of  R_i,j
                                             
          - determine output value for each leaf
          - leaf R_2,1 has two output values
          - the output value for each leaf is the value for γ that mimimizes this summation
                γ_j,m = argmin_γ ∑ L(y_i,F_m-1(x_i) + γ)   for x_i element of  R_i,j

           - calculate R_1,1:
             - replace generic Loss Function: L(y_i,F_m-1(x_i) + γ)
                  with actual loss function  1/2(observed - predicted)**2 
                  that is : 1/2 (y_i - (F_m-1(x_i) + γ))**2
                  expand the summation into individual terms
                     ∑ L(y_i,F_m-1(x_i) + γ)
             - j=1 for first leaf,  and m=1 for first tree
                 γ_1,1 =  argmin_γ  ∑ ((y_i - (F_m-1(x_i) + γ))**2
                        Notes: y_i: observed value, F_m-1(x_i): predicted value, F_m-1(x_3) = 73.3
                       γ_1,1 = argmin_γ  1/2(56 - 73.3 - γ)**2  
                             = argmin_γ  1/2(-17.3 - γ)**2  
              - find the value for γ that minimizes this equation
                 - take the derivative of the Loss function with respect to γ

                 (d/dγ) (1/2) (-17.3 - γ)**2  -> 
                     apply chain rule a set derivative to 0 ->    17.3 + γ = 0 
                     -> the value for gamma that minimizes this equation is: -17.3
                     γ_1,1 = 17.3
                 
           - calculate R_2,1:
               
                γ_2,1 = argmin_γ ∑ L(y_i,F_m-1(x_i) + γ)   for x_i element of  R_i,j
             plug in loss function 1/2(observed - predicted)**2 
                γ_2,1 = argmin_γ ∑ 1/2(y_i - (F_m-1(x_i) + γ))**2 
             expand summation
                γ_2,1 = argmin_γ [ 1/2(y_1 - F_m-1(x_1) + γ)**2  +  1/2(y_2 - (F_m-1(x_2) + γ))**2 ]
             plug in observed weights
                γ_2,1 = argmin_γ [ 1/2(88 - (F_m-1(x_1) + γ))**2  +  1/2(76 - (F_m-1(x_2) + γ))**2 ]
             plug in '73.3' for F_m-1(x_i) since F_0(x) = 73.3
                γ_2,1 = argmin_γ [ 1/2(88 - (73.3 + γ)**2  +  1/2(76 - (73.3 + γ))**2 ]
             simply
                γ_2,1 = argmin_γ [ 1/2(14.7 - γ)**2  +  1/2(2.7 - γ)**2 ]

              - find the value for γ that minimizes this equation
                 - take the derivative of the Loss function with respect to γ

                 (d/dγ) (1/2) [ 1/2(14.7 - γ)**2  +  1/2(2.7 - γ)**2 ]
                     apply chain rule a set derivative to 0 ->    -14.7 + γ + -2.7 + γ = 0 
                     -> the value for gamma that minimizes this equation is: 
                     γ_2,1 = (14.7 + 2.7) / 2  = 8.7
                     -> end up with the average of the residuals that ended in leaf R_2,1
                     
                 
                      Height<1.55    
                      /        \    
                  -17.3       14.7,2.7
                  R_1,1         R_2,1      
             γ_1,1 = -17.3     γ_2,1 = 8.7

             Note: given our choice of the Loss Function, the output values are ALWAYS the average
                   of the Residuals that end up in the same leaf



       D. Update F_m(x) = F_m-1(x) +  η  ∑ [γ_m I(x element of R_jm)]  for j=1 to J_m

           - F_m(x) is a new prediction for each sample
           - this new prediction is F_1(x) and is based on the last prediction F_0(x)
             plus the tree when just finished
                 F_1(x) = F_0(x) +  η  ∑ [γ_m I(x element of R_jm)]  for j=1 to J_m
       
           - The summation says we should add up the Output values γ_j,m 's for all the leaves,
             R_j,m , that a sample, x can be found in 

           - "eta" (η) is the learning rate and is a value between 1 and 0
              - a small Learning rate reduces the effect each tree has on the final prediction, 
                and this improves the accuracy in the long run
                - set eta to 0.1

            - predictions for each sample, x_1, x_2, & x_3: 
                 F_1(x_1) = 73.3 +  0.1 * 8.7   = 74.2
                 F_1(x_2) = 73.3 +  0.1 * 8.7   = 74.2
                 F_1(x_3) = 73.3 +  0.1 * -17.3 = 71.6



          height   Favorite    Gender   Weight    r_i,1  predicted
           (m)      color                (kg)            weights r_i,1
        ----------------------------------------  ------ ---------
           1.6       Blue        Male     88       14.7    74.2
           1.6       Green      Female    76        2.7    74.2
           1.5       Blue       Female    56      -17.3    71.6


   Step 2 review
         - started by setting m=1
       A. Compute r_i,m = -[ ∂L(y_i,F(x_i)) / ∂F(x_i) ] for F(x)=F_m-1(x)  for i = 1,...,n
          - solved for the negative gradient 
          - plugged in the observed values for y_i
          - plugged in the latest predictions for F(x_i)
          - this gave us residuals

       b. Fit a regression tree to r_im values and create terminal regions R_j,m for j = 1 ... J_m
         - fitted regression tree to the residuals (that is, create a new tree)

                      Height<1.55    
                      /        \    
                  -17.3       14.7,2.7
                  R_1,1         R_2,1      
             γ_1,1 = -17.3     γ_2,1 = 8.7

       c. For j = 1...J_m compute  γ_j,m = argmin_γ ∑ L(y_i,F_m-1(x_i) + γ)   for x_i element of  R_i,j
         - computed the output values, γ_i,1 for each leaf

                      Height<1.55    
                      /        \    
                  -17.3       14.7,2.7
                  R_1,1         R_2,1      
             γ_1,1 = -17.3     γ_2,1 = 8.7


       D. Update F_m(x) = F_m-1(x) +  η  ∑ [γ_m I(x element of R_jm)]  for j=1 to J_m
          - made new predictions, F_1(x), for each sample based previous prediction, F_0(x) = 73.3,
            the learning rate (η = 0.1) and the output values  γ_1,1 and  γ_2,1  from the new tree

   Next, set m = 2, and do repeat step 2

      tree 2:
                      Height<1.55    
                      /        \    
                  -15.6       13.8,1.8
                  R_1,2         R_2,2      
             γ_1,2 = -15.6     γ_2,2 = 7.8

   Assume M=2:

   Step Output F_M(x)  + 0.1 x tree1 residuals  +  0.1 x tree2 residuals
      F2(x) = 73.3 + 0.1 x 

--------------------------------------------------
Gradient Boost Part 3 (of 4): Classification
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=jxuNLH5dXCs

  Summary:
    - This is Part 3 in our series on Gradient Boost. At long last, we are showing how it can be used for classification. 
    - This video gives focuses on the main ideas behind this technique. 


  Dataset:

      Likes   Age   Favorite  Loves 
     Popcorn          Color   Troll 2
     ---------------------------------
       Yes      12     Blue    Yes
       Yes      87     Green   Yes
       No       44     Blue    No 
       Yes      19     Red     No 
       No       32     Green   Yes
       No       14     Blue    Yes


  Gradient Boost for Classification
    - initial Prediction for every individual is the log(odds)
    - log(odds) is roughly equivalent to the Logistic Regression of average
    - the easiest way to use the log(odds) for classification is to convert it to a probability
      and we use the "logistic Function":
           probability of target = e**log(odds) / (1 + e**log(odds))


    Calculate prediction using log(odds)
        loves Troll 2: log(4/2) = 0.7   <-  initial prediction

    Calculate the probability using the logistic function
           probability of loving Troll2 = e**log(odds) / (1 + e**log(odds))
                                        = e**log(4/2)  / (1 + e**log(4/2))  = 0.7

           Note: logs(odds) and probability are the same only because of rounding.
               e.g.   log(4/2)                           = 0.6931
                       e**log(4/2)  / (1 + e**log(4/2))  = 0.6667
                      

    Initial prediction
       - since the probability of "loving troll 2" is greater that 0.5, we can classify
         everyone in the Training Dataset as someone who "loves troll 2"
       - Note: while 0.5 is a very common threshold for making classification decisions based on 
               probibility, we could have just as easily used a different value

    Calculate Residual
       - measure how bad the initial prediction by calculating the "pseudo presiduals", the
         difference between  the observed and the predicted
           residual = (observed - predicted)
       - observed values:  1: loves troll2 0,   0: does not love troll2, predicted value (probability): 0.7 
              residual loves troll2 - yes = 1 - 0.7 =  0.3
              residual loves troll2 - no  = 0 - 0.7 = -0.7



    Build Tree;
      - just like Gradient boost for Regression, we are limiting the number of leaves that we will
        allow in the the tree. For simple example, limiting leaves to 3
      - in practice, the maximum number of leaves is often set between 8 and 32


                      Color=Red
                     T/        \F   
      residual -> -0.7       Age > 37
      output ->  -3.3      /          \
                      0.3,-0.7       0.3,0.3,0.3     <- residuals
                         -1             1.4          <- Outputs

      Handling multiple output values
        - the predictions are in terms of the log(odds)
        - the leaf is derived from a probability
        - so we can't just add multiple outputs together to get a new log(odds) prediction without
          some sort of transformation
        - When using Gradient Boost for Classificaiton, the most common transformation formula is:
           - Note: formula derivation is covered in part 4
           - Note: previous probabilities are all the same for all of the Residuals, but this will 
             change when we build the next tree

               output value = ∑ Residual_i  / ∑ [PreviousProbability_i x (1 - PreviousProbability_i)] 

             leaf 1 transformation to calcuate new output value:
                 -0.7  / (0.7 x (1 - 0.7)) = -3.3  

             leaf 2 transformation to calcuate new output value:
                (0.3 + -0.7)  / [(0.7 x (1 - 0.7)) + (0.7 x (1 - 0.7)) ] = -0.4 / 0.42 = -1  

             leaf 3 transformation to calcuate new output value:
                (0.3 + 0.3 + 0.3)  / [(0.7 x (1 - 0.7)) + (0.7 x (1 - 0.7)) +  (0.7 x (1 - 0.7)) ]  = 0.9 / 3 x 0.21 = 1.4

        Update predictions
          - combine initial leaf with new tree
          - scaled by a learning rate 
            - this example uses a relatively large learning rate, 0.8, for illustrative purposes. 
              0.1 is a more common learning rate

           log(odds) prediction = previous prediction + learning rate x output

           sample 1, 5, 6
              log(odds) prediction = log(4/2) + 0.8 x 1.4 = 1.8 
              convert to probability: e**1.8 / (1 + e**1.8) = 0.9
                
           sample 2, 3
              log(odds) prediction = log(4/2) + 0.8 x -1 = -0.1 
              convert to probability: e**-0.1 / (1 + e**-0.1) = 0.5
                
           sample 4
              log(odds) prediction = log(4/2) + 0.8 x -3.3 = -1.8 
              convert to probability: e**-1.8 / (1 + e**-1.8) = 0.1
                
        Update residuals
           residual = (observed - predicted)

           samples 1, 5, 6: residual = (1.0 - 0.9) =  0.1
           samples 2      : residual = (1.0 - 0.5) =  0.5
           samples 3      : residual = (0.0 - 0.5) = -0.5
           samples 4      : residual = (0.0 - 0.1) = -0.1

    Build Tree 2;
                         Age < 66
                     T/             \F   
                 Age > 37            0.5            <- residual
               /       \              2             <- output 
            -0.5     0.1,-0.1,0.1,0.1               <- residuals
             -2           0.6                       <- Outputs


     Calculate tree 2 output values:
             output value =  ∑ Residual_i  / ∑ [PreviousProbability_i x (1 - PreviousProbability_i)] 

             leaf 2 transformation to calcuate new output value:
                 0.5  / (0.5 x (1 - 0.5)) = 2

             leaf 3 transformation to calcuate new output value:
                -0.5  / (0.5 x (1 - 0.5)) = -2

             leaf 1, 4, 5 , 6 transformation to calcuate new output value:
                (0.1 + -0.1 + 0.1 + 0.1)  / [(0.9 x (1 - 0.9)) +  (0.1 x (1 - 0.1)) + (0.9 x (1 - 0.9)) +  (0.9 x (1 - 0.9)) ] 
                   = 0.2 / [3((0.9 x (1 - 0.9)) +  (0.1 x (1 - 0.1))]  = 0.2 / 0.27 + 0.09 = 0.6


          Likes   Age   Favorite  Loves     Residual  Predicted   Residual
         Popcorn          Color   Troll 2    initial  Probability    1
         ---------------------------------   -------  --------     -------
           Yes      12     Blue    Yes         0.3      0.9          0.1
           Yes      87     Green   Yes         0.3      0.5          0.5
           No       44     Blue    No         -0.7      0.5         -0.5
           Yes      19     Red     No         -0.7      0.1         -0.1
           No       32     Green   Yes         0.3      0.9          0.1
           No       14     Blue    Yes         0.3      0.9          0.1


  Gradient Boost for Classification steps review:
     one leaf prediction
       - started with just a leaf, which made one prediction for every individual
         log(4/2) = 0.7
     initial tree
       - build a tree based on the 'residuals', the difference between the 'observed' values and
         the single value 'predicted' by the leaf
           residual = (observed - predicted)
      calculate output values for each leaf tree 1
           output value =  ∑ Residual_i  / ∑ [PreviousProbability_i x (1 - PreviousProbability_i)] 
       calcuate predicted probabiliy
           log(odds) prediction = previous prediction + learning rate x output
       calcuate residual 1
           residual = (observed - predicted)
       Build tree 2 based on the new residuals

       Repeat
         - this process repeats until we have made the maximum number of trees specified of the residual
           gets super small

       Make Predictions (assume max 2 trees for example)
         
       
          Likes   Age   Favorite  Loves     
         Popcorn          Color   Troll 2   
         --------------------------------- 
           Yes      25     Green   ???     

             Log(odds) Prediction that some Loves troll 2: 
                                                = initial pred. + (η x tree1 output) + (η x tree2 output)
                                                 = 0.7 +          (0.8 x 1.4)        + (0.8 x 0.6) = 2.3

             convert log(odsd) to probibility:
                    probability of loving Troll2 = e**log(odds) / (1 + e**log(odds))
                                                 =  e**2.3 / (1 + e**2.3) = 0.9

              classify:
                 - since 0.9 > 0.5, we will classify this person as "Yes" Loves Troll 2


--------------------------------------------------
Gradient Boost Part 4 (of 4): Classification Details
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=StWY5QWMXCw

  Summary:
    - At last, part 4 in our series of videos on Gradient Boost. 
    - This time we dive deep into the details of how it is used for classification, going through algorithm, 
      and the math behind it, one step at a time. 
    - Specifically, we derive the loss function from the log(likelihood) of the data and we derive the functions 
      used to calculate the output values from the leaves in each tree




--------------------------------------------------
Linear Regression, Clearly Explained!!!
  https://www.youtube.com/watch?v=nk2CQITm_eo

  Summary: 
    - statquest: Linear Regression (aka General Linear Models, part 1)
    - The concepts behind linear regression, fitting a line to data with least squares and R-squared, are 
      pretty darn simple, so let's get down to it! 
    - NOTE: This StatQuest comes with a companion video for how to do linear regression in 
      R: • Linear Regression in R, Step-by-Step  
      https://www.youtube.com/watch?v=u1cc1r_Y7M0
    - You can also find example code at the StatQuest github: 
         https://github.com/StatQuest/linear_regression_demo
          


  Linear Regression Main Ideas
    1. Use least-squares to fit a line to the data
    2. Calcuate R**2
    3. Calcuate a p-value for R**2

    Dataset: 
       - samples of mouse weight and mouse size
       - use mouse weight to predict mouse size

    Steps:
       1. draw a line through the data
       2. measure the distance from the line to each data point, square the distance, and add them up
       Residual
         - the distance from a line to a data point is call a 'residual'
       3. rotate the line a little
         - with the new line, measure teh residuals, square them, and then sum up the squares
       4. repeat 3 a number of times
       5. plot the sum of squared residuals (y-axis) and corresponding rotations (x-axis) 
         - find the rotation that has the least sum of squares - this fits the data the best
       Least Squares
         - method for fitting a line to the data

    R squared
      - first, calculate the average mouse size
      - next, sum the squared residuals of the each mouse size and the average size
      - tells us how much of the variation in mouse size can be explained by taking mouse weight into account
      R**2  = [Var(mean) -  Var(fit)] /  Var(mean)
         or
      R**2  = [SS(mean) -  SS(fit)] /  SS(mean)

      In the mouse weight & size example:
            Var(mean) = 11.1        SS(mean) = 100
            Var(fit)  = 4.4         SS(fit)  = 40

            R**2  = [Var(mean) -  Var(fit)] /  Var(mean)
                  = [11.1 - 4.4]  / 11.1 = 0.6

              - this means: there is a 60 % reduction in the variance when we take the mouse weight into account
              - alternative, we can say that moust weight "explains" 60% of the variation in mouse size

            R**2  = [SS(mean) -  SS(fit)] /  SS(mean)
                  = [100 - 40] / 100  = 0.6 = 60%

      For target data around the target average  (mouse size only)
        SS(mean) 
          - means: Sum of Squares around the mean 
          SS(mean0 = (data - mean)**2
        Var(mean)
          - variation around the mean 
          - another way to think about  variance is as the average sum of squares per mouse
            Var(mean) = (data - mean)**2  / n     where n: sample size
                      = SS(mean) / n

      For sample/target data distance from linear line
       SS(fit)
         - sum of squares around the least-squares fit
         - for between linear line and data points (i.e. mouse weight and mouse size) 
         SS(fit) = (data - line)**2

        Var(fit) 
          - the variation around the fit
          Var(fit) = (data - line)**2  / n     where n: sample size
                   = SS(fit) / n
       
         In general:
            Variance(something) =  Sums of squares / number of those things = average of sum of squares

    3D case: 
       - we want to know if the mouse weight and tail length do a good job of predicting the mouse length
       - plot on 3D graph (x-axis: weight, z-axis: tail length, y-axix body length
       - do a least-square fit. Since extra dimension (term), fit a plane instead of a line
       - plane equation: 
         y = 0.1 0.7x + 0.5z
           - least-square estimates 3 parameters (y-intercept, mouse weight, & mouse tail length)
        - measure residuals and calculate R**2  : (observed - predicted)**2
        - if the tail length is useless, and doesn't make SS(fit) smaller, then least-square will ignore
          it by making that parameter = 0

    Adjusted R**2
      - adjusted R**2 scales R**2 by the number of parameters
      - the more parameters we add to the equation, the more opportunities we have for random events to
        reduce SS(fit) and result in a better R**2

    R**2 Review:
       R**2 = [Var(mean) - Var(fit)] / Var(mean)
            = [Var(mouse size) - Var(after taking weight into account) / Var(mouse size)
            = the variation in mouse size explained by weight / variation in mouse size with taking weight into account

    p-value
       - need a way to determine if the R**2 value is statistically signficant
         - for example, if only 2 datapoints, R**2 = 100% which does not mean anything

       - the p-value for R**2 comes from something called "F"

            F = the variation in mouse size explained by weight / variation in mouse size not explained by weight

            F = [SS(mean) - SS(fit) / (p_fit - p_mean)] /  [SS(fit) / (n - p_fit)} 

             
             These numbers are the "degrees of freedom:
                                    / (p_fit - p_mean)] /           / (n - p_fit)} 
                  - they turn the sums of squares into variances
                  - another statquest video on "degrees of freedom

                   p_fit:  the number of parameters in the fit line  (y = y-intercept + slope x  -> has 2 parameters)
                   p_mean: the number of parameters in the mean line  (y = y-intercept           -> has 1 parameter)

                   (p_fit - p_mean) = p_extra:  (in example: 2 - 1 = 1)
                      - the variance explained by the extra parameter. 
                      - in our example, that's the variance in mouse size explained by mouse weight

                    (n - p_fit)
                      - intuitively, the more parameters you have in your equation, the more data
                        you need to estimate them
                        - for example, you only need 2 points to estimate a line, but you need 3 points
                          to estimate a plane


                                                                          if "fit" is good:
                the variation explained by extra parameters in the "fit"           large number
            F = ---------------------------------------------------------   -->    ------------
                the variation NOT explained by extra parameters in the "fit"        small number

              if fit is "good", then:  F = large number

           conceptually Calculate p-value from "F"
             - the p-value is the number of more extreme values divided by the all the values
             - see statquest p-value video
              conceptally:
                 repeat large number of times:
                 - generate a set of random data
                 - calculate the mean and SS(mean)
                 - calculate the "fit" and SS(fit)
                 - plot on histogram


               F = [SS(mean) - SS(fit) / (p_extra)] /  [SS(fit) / (n - p_fit)} 


           Calculate p-value from "F"
              - you can approximate the histogram with a line
              - in practice, rather than generating tons of random datasets, people use the line to 
                calculate the p-value
              - the p-value will be smaller when there are more samples relative to the number of parameters
                in the fit equation
              
   Main ideas:
      - given some data that you think are related, linear regression:
        1. Quantifies the relationship in the data (this is R**2)
          1) this needs to be large
        2. Determines how reliable that relationship is (this is the p-value that we calcuated with "F"
          1) this needs to be small

--------------------------------------------------
Linear Regression in R, Step-by-Step
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=u1cc1r_Y7M0

  Summary:
    - This video, which walks you through a simple regression in R, is  a companion to the StatQuest 
      on Linear Regression    • Linear Regression, Clearly Explained!!!  
    - If you want to just copy and paste the R code, you can get it from the StatQuest GitHub site: 
         https://github.com/StatQuest/linear_regression_demo

  Code: Linear Regression R code

    ## Here's the data from the video
    mouse.data <- data.frame(
      weight=c(0.9, 1.8, 2.4, 3.5, 3.9, 4.4, 5.1, 5.6, 6.3),
      size=c(1.4, 2.6, 1.0, 3.7, 5.5, 3.2, 3.0, 4.9, 6.3))
    
    mouse.data # print the data to the screen in a nice format
    
    ## plot a x/y scatter plot with the data
    plot(mouse.data$weight, mouse.data$size)
    
    ## create a "linear model" - that is, do the regression (size; y-values, weight: x-values)
    #  size = y-intercept + slope x x-values
    # linear model calculates the least-square estimate for the y-intercept and slope
    mouse.regression <- lm(size ~ weight, data=mouse.data)
    ## generate a summary of the regression
    summary(mouse.regression)
    # --- summary output ----

    call: 
    lm(formulaa = size ~ weight, data = mouse.data)    <--- prints out the original call to lm() function
    
    Residuals:
       Min       1Q  Median      3Q    Max       <--- Summary of the residuals (distance from the data to the
    -15.482 -0.8037  0.1186  0.6186 1.8852            fitted line. Ideally, they should be symmetrically 
                                                      distributed around the line
        
    Coefficients:                                   <--- tells about the least-squares estimates for the fitted data
               Estimate Std. Error t value  Pr(<}t})
    (Intercept)  0.5813     0.9647   0.603  0.5658
    weight       0.7778     0.2334   3.332  0.0126 *
    ---
    Signif. codes: 0 '***' 0.001 "**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 
    
    Residual standard error: 1.19 on 7 degrees of freedom  <--- square root of the denominator in the equation for 'F' 
    Multiple R-squared: 0.6133,  Adjusted R-squared: 0.558
    F-statistic:  11.1 on 1 and 7 DF, p-value 0.0.1256
    # --- end summary output ----

    ## add the regression line to our x/y scatter plot
    abline(mouse.regression, col="blue")



  Symmetrically distributed around the line:
    - want the min and max values to be approximately the same distance from the line
    - want the 1Q (1st quantile) and 3Q (3rd Quartile) values to be approximately the same distance from the line
    - have median close to zero


  Least-squared estimates:
    - includes the y-intercept (0.5813), slope (0.7778), 
    - std errors and t-values are provided to show how the P-value was calculated
    - p-values test whether the estimates for the intercept and the slope are equal to zero or not.
      if they are equal to zero, they don't have much use in the model
    - the p-values (Pr(>|t|) column) for the estimated parameters
      - generally, not usually interested in the intercept, so it does not matter what its p-value is.
      - want the p-value for "weight" to be < 0.05. That is, we want it to be statistically signficant 
        (give a reliable most size)

               Estimate Std. Error t value  Pr(>|t|)
    (Intercept)  0.5813     0.9647   0.603  0.5658
    weight       0.7778     0.2334   3.332  0.0126 *

       0.5813 -> intercept
       0.7778 -> slope


  Multiple R-squared
    - just R-squared described in previous video
    - value 0.6133 means the weight explains 61% of the variation in size 

  Adjusted R-squared
    - R-squared scaled by the number of parameters in the model

  F-statistic:
   - line provided info on whether R-squared is significant
      - F-value: 11.1,   degrees of freedom: 1 and 7, and p-value: 0.01256
      - this p-value indicates that weight gives a reliable estimate for size


--------------------------------------------------
Multiple Regression, Clearly Explained!!!
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=zITIFTsivN8

  Summary:
   - This video directly follows part 1 in the StatQuest series on General Linear Models (GLMs) on Linear 
      Regression • Linear Regression, Clearly Explained!!!   
   - This StatQuest shows how the exact same principles from "simple" linear regression also apply multiple regression. 
   - At the end, I show how to test if a multiple regression is better than a simple regression


  Simple Regression
    - fitting a line to the data (y = y-intercept + slope x)
    - interested in the R-squared and the p-value to evaluate how well the line fits the data

  Multiple Regression
     - for multiple regressions, you fit a plane (for 3D) or some higher-dimensional object to your data
     - for plane: y = y-intercept + x-slope x + z-slope Z
     - calculating R-square si the same for simple and multiple regressions

      R-squared  = [SS(mean) -  SS(fit)] /  SS(mean)

        where: 
          SS(mean) 
            - means: Sum of Squares around the mean 
            SS(mean0 = (data - mean)**2


         SS(fit)
           - sum of squares around the least-squares fit
           - for between linear line and data points (i.e. mouse weight and mouse size) 
           SS(fit) = (data - line)**2

      Calculating the p-value is the same:
       - the p-value for R**2 comes from something called "F"

            F = the variation in mouse size explained by weight / variation in mouse size not explained by weight

            F = [SS(mean) - SS(fit) / (p_fit - p_mean)] /  [SS(fit) / (n - p_fit)} 

              p_fit
                - for simple regression, p_fit = 2
                - for multiple regression, p_fit = 3  or greater
              p_mean
                - p_mean = 1 for both simple and multiple regression, because we only need to
                  estimate the mean value for the target (i.e. body length)

   Compare Simple Regression Fit vs Multiple regression filt
     - this tells us if there is value in the extra parameter(s) used in the multiple regression
     - compare the simple regression fit vs multiple regression fit

            F = [SS(simple_fit) - SS(multiple_fit) / (p_multiple - p_simple)] /  [SS(multiple_fit) / (n - p_multiple)} 

     - if the difference in the R-squared values between the simple and multiple regressions is
       "big" and p-value is small, then adding the extra parameter(s) is worth it

--------------------------------------------------
p-values: What they are and how to interpret them
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=vemZtEM63GY

  Summary:
    - This StatQuest is all about interpreting p-values. 


  p-values:
    - numbers between 0 and 1 that quantify how confident we should be that item1 (e.g. Drug A)
      is different from item2 (e.g. Drug B)
    - the closer to 0, the more confidence we have that item1 (e.g. Drug A) and item2 (e.g. Drug B)
      are different
    - a commonly used threshold is 0.05. 
        - it means that if there is no difference between item1 and item2, and we did this exact 
          same experiment a bunch of times, then on 5% of those experiments would result in a 
          wrong decision
        - used because reducing the number of 'false positives' below 5% oftens costs more that
          it is worth
        - means that if the p-value < 0.05, then we will decide that item1 (e.g. Drug A) is different 
          than item2 (e.g. Drug B)

    Exeriment - give Drug A to two different grougs:

          Group 1 - Drug A              Group 2 - Drug A              
          Cured     Not Cured   p=0.9   Cured     Not Cured
            73       125                 71        127

        - the p-value calcuated usign "Fisher's Exat Test" is 0.9
        - if repeated, each experiment should get similarly large p-values, but once in a while, different results
        false positive
          - getting a small p-value when there is no difference is called a 'false positive'

  Hypothesis testing
    - the null hypothesis is that the drugs are the same
    - the p-value helps us decide if we should reject the 'Null Hypothesis' or not


  p-value interpretation
    - while a small p-value helps us decide if item1 is different from item2, it dos NOT
      tell how different they are
    - that is, you can have a small p-value regardless of the size of the difference between
      item1 and item2. The difference could be tiny or huge

--------------------------------------------------
How to calculate p-values
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=JQc3yx0-Q9E

  Summary:
    - In this StatQuest we learn how to calculate p-values using both discrete data (like coin tosses) and 
      continuous data (like height measurements). 
    - At the end, we explain the differences between 1 and 2-sided p-values and why you should avoid 
      1-sided p-values if possible.
    

  p-value types:
    one-sided
      - rarely used, and potentially dangerous, so they should be avoided
    two-sided
      - most common, and this statquest focuses on how to calculate

  Null Hypothesis:
    example: Even though I got 2 Heads (of coin toss) in a row, my coin is no different from 
             a normal coin
     - if we reject this 'null hypothesis', we will know that our coin is special
     - test hypothesis with p-values


  p-values
    - are determined by adding up probabilities, 
    - so let's start by figuring out the probability 

    coin flip twice outcomes (assume 50% for heads and 50% for tails):
        outcomes 
        1     2    probability
      head  head     0.25
      head  tail  <-|0.50
      tail  head  <-|
      tail  tail     0.25

  p-value is composed of 3 parts:
    1. the probability random chance would result in the observation    (e.g. 2 heads: 0.25)
    2. the probability of observing something else that is equally rare  (e.g. 2 tails: 0.25)
      - we add part 2, the probability of something else that is equally rare, because alto getting
        2 heads might seem special, it doesn't seem as special when we know that other things
        are equally rare
    3. the probability of oberving something rarer or more extreme       (e.g. 0 in this example)
       - in this cases, this is zero because no other outcomes are rarer that 2 heads or 2 tails
       - because rarer thing make something less special, we add part 3 to the p-value

       p-value for 2 heads = 0.25 + 0.25 + 0 = 0.5
        - typically, we only reject a hypothesis is the 'p-value' is less than 0.05

  Null Hypothesis:
    example: Even though I got 4 Heads (of coin toss) and 1 tail, my coin is no different from 
             a normal coin
 
   5 toss outcomes
     - 32 possible outcomes
     1. the probability that we randomly get 4 heads and 1 tail:  5/32
     2. the probability of observing something else that is equally rare 5/32 (for 4 Tails & 1 Head) 
     3. the probability of oberving something rarer or more extreme       (e.g. 0 in this example)
        5 Heads 1/32  + 5 tails 1/32 = 2/32

        p-value for 4 heads an 1 tail = 5/32 + 5/32 + 2/32 = 0.375
         - reject null hypothesis - our coin is not special

   p-values for continuous values
     - we calculate probabilities and p-values for something continuous, like height, we usually
       us something called a statistical distribution

    Example: distribution of height measurement from Brazilian women between 15 and 49 years old from 1996
       - graph show a blue colored bell-like curve with 2 std deviation at 142 cm (4.5 ft) and 
         169 cm (5.5 ft) and average: 155.7cm 
       - 95% of the area under the curve is between  142 cm (4.5 ft) and 169 cm (5.5 ft)
          - means 95% probability that when we measure a brazilian woman, their height will be 
            between  142 cm (4.5 ft) and 169 cm (5.5 ft)
           - 2.5% of the total area under the curve is less than 142, so 2.5% chance their
             measured height will be less than 142cm

   p-values with a distribution
     - to calculate p-values with a distribution, you add up the percentages of the area under the curve


   Example: measured height is: 142 cm
       is the 142cm measurement so far from the mean of the distribution (blue colored bell curve)
        from the mean (155.7cm) that we can reject the idea that it came from it?

        p-value for 142cm
            - add area for people less than or equal to 142 cm (0.025)
            - Note: when working with distributions, we are interested in adding more extreme values
               to the p-value rather than rarer values
            - add area for people greater than or equal to 169 cm (0.025)
               - these values are considered equal to more more extreme because they are as far from 
                 the mean (155.7 cm), or further

        p-value for 142cm = 0.025 + 0.025 + 0.05
          - since the cut-off is 0.05, maybe it could come from this distribution, maybe not. 
            It's hard to tell since the p-value is on the borderline

   Example: measured height is: between 155.4 cm and 156 cm

         p-value = 0.04 + 0.48 + 0.48 = 1
                   0.04: probability between 155.4 cm and 156 cm
                   0.48: probability that height is less than 155.4
                   0.48: probability that height is greater than 156

              - means that, given this distribution of heights, we would not find it unusual to
                measure someone who's height was close to the average even though the  probability
                is small (0.04)


   1-sided p-values
     - previous calculations were all for 2-sided p-values
     - why it has the potential to be dangerous
       - only measures more extreme values in one direction
       For one-sided p-value, we only use the area that is in the direction we want to see change

  in summary:
    p-value composed of three parts:
    1. the probability random chance would result in the observation   
    2. the probability of observing something else that is equally rare 
    3. the probability of oberving something rarer or more extreme  


--------------------------------------------------
What are degrees of freedom
  James Gilbert
https://www.youtube.com/watch?v=rATNoxKg1yA


  Degrees of Freedom Definition 
    - number of independent pieces of information to make your calculation (e.g. mean calculation)
    - not the same as the sample size

   Coin toss
     - has 1 degree of freedom since you know you can only get heads or tails
     - only 1 independent piece of information required (if toss tails, you also know you did not get heads)

   traffic color (red, yellow, or green)
     - two pieces of information are needed (e.g. not yellow, not green)
     - has 2 degrees of freedom

   Degrees of Freedom
     - usually, the number of categories - 1 (DF = k - 1)


   Mean of a sample
       Formula: x-mean = ∑ x_i  / N 
          -> can change one of the sample values, can change the mean value

   Std Deviation
       S = √ [ ∑ (x_i  - x-mean)**2  / (N - 1)]

       - does not depend on all your sample values, only 'N - 1'
       - because you already worked out the mean, the N value does not contribute the std deviation
       - if you already know the mean, then you 
       - degrees of freedom is N - 1


--------------------------------------------------
Using Linear Models for t-tests and ANOVA, Clearly Explained!!!
  StatQuest with Josh Starmer

  Summary:
    - this StatQuest shows how the methods used to determine if a linear regression is statistically significant 
      (covered in part 1) can be applied to t-tests and ANOVA. 
    - It also introduces the concept of a "design matrix". 


  in part 1, measure mouse weight and mouse size, and we wanted to learn 2 things:
    1. how useful was the mouse weight for predicting the mouse size (R-squared told us this)
    2. Was that relationship due to change? (p-value told us this)

  t-test
    - t-test goal is to compare means and see if they are significantly different from each other
    steps:
    1. Ignore the X-axis and find the overall mean
    2. calculate the Sum of Square residuals around the mean, SS(mean). 
      - residuals: the distance from the data points to the line, in this can the mean line
    3. Fit a line to the data
      - start by finding a least squares fit to the control data
        - it turns out that the mean is the least-squares fit
        - the mean intercepts the y-axis at 2.2 (mean: y = 2.2; horizontal line)
      - next find the least squares fit to the mutant data ( y = 3.6)
      - combine the control mean line and the mutant mean line into 1 line

        y = 1 x 2.2 + 0 x 3.6 + residual   <- for control dataset

        y = 0 x 2.2 + 1 x 3.6 + residual   <- for mutant dataset

        design matrix
          - design matrix can be combined with an abstract version of the equation to represent a "fit"
            to the data

        y = column1 x 2.2 + column2 x 3.6 + residual   <- column1 turns the control mean "on" and "off"
                                                       <- column2 turns the mutant  mean "on" and "off"

          - in practice, the role of each column is assumed, and the equation is written                                               

           y = mean_control + mean_mutant

     4. Calculate SS(fit), the sum of squares of the residuals around the fitted line(s)
       - SS(fit) for t-test is the sum of squared residuals


            F = [SS(mean) - SS(fit) / (p_fit - p_mean)] /  [SS(fit) / (n - p_fit)} 


            p_mean = 1
              - just 1 parameter in the mean gene expressions
            p_fit
              - the number of parameters in the that we fitted for the t-test,
                That is:   y = mean_control + mean_mutant  <--- 2 parameters
                    - one parameter for the mean of the control data
                      and one parameter for the mean of the mutant data

  t-test Review
    1. calcuated sum of squares of the residuals around the overall mean, SS(mean)
       y = overall mean
    2. calcuated sum of squares of the residuals around the fit, SS(fit)
      - to do this with a single equation, needed to create a design matrix
           y = mean_control + mean_mutant
    3. plug values into "F equation" to get our p-value
           p_mean =1, p_fit =2
           F = [SS(mean) - SS(fit) / (p_fit - p_mean)] /  [SS(fit) / (n - p_fit)} 

   ANOVA
     - test if all 5 categories are the same
     example: contol mice, mutant mice, control mice and mutant mice on funky diet, heterozogote

    1. calcuated sum of squares of the residuals around the overall mean, SS(mean)
       - overall mean which includes all 5 categories
       y = overall mean 
    2. calcuated sum of squares of the residuals around the fit, SS(fit)
      - to do this with a single equation, needed to create a design matrix
           y = mean_control + mean_mutant + mean_funky_control + mean_funky_mutant + mean_heterozygote
    3. plug values into "F equation" to get our p-value
           p_mean =1, p_fit =5
           F = [SS(mean) - SS(fit) / (p_fit - p_mean)] /  [SS(fit) / (n - p_fit)} 


--------------------------------------------------
Odds and Log(Odds), Clearly Explained!!!
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=ARfXDSkQf1Y

  NOTE: log is actual log_e = ln

  Summary:
    - The odds aren't as odd as you might think, and the log of the odds is even simpler! 
    - This StatQuest covers those subjects so that you can understand the statistics for true/false type problems 
      (like Logistic Regression).
      - alternatively, odds of winning = probabiliity of winning / [1 - probability of winning]
      - calculating odds from probabilities:
           odds of winning = probability of winning / probability of losing 
                           = probability of winning / [1 - probability of winning] 
                           simplified to: p / (1 - p)


  Odds
    - odds are the ratio of something happening (i.e. team A winning) to something not happening
      (i.e. team A not winning)
    - odds are NOT probabilities

  Probabilities
    - probalities are the ratio of something happening (i.e. team A winning) to everything that happen
      (i.e. team A winning and not winning)
    - alternatively, the probality of losing equals 1 - probability of winning 


   example:  Team A will win 5 games and loss 3 games

       odds of winning = 5/3 = 1.7

       probability of winning = 5 / [3 + 5] = 5/8 = 0.625

       probability of lossing = 3 / [3 + 5] = 5/8 = 0.375

       probability of lossing = 1 - probability of winning = 1 -  3 / [3 + 5] = 1 - 5/8 = 0.375

       odds of winning = probability of winning / probability of losing =    [5/8] / [3/8]  = 5/3 = 1.7 


  log of the odds
    - odds of team A winning (i.e. with a winning records) are from 1 to infinity 
    - odds against of team A winning (i.e. with a lossing records) are from 1 to 0 
    - winning records vs lossing record produce unsymmetrical results:

        example: odds with 6 to 1 record and with 1 to 6 record
            6 to 1 odds:  6/1 = 6
            1 to 6 odds:  1/6 = 1/6
              - unsymmetrical results

     - log() of the odds (i.e. log(odds)) solves this problem by making everying symmetrical
        example: odds with 6 to 1 record and with 1 to 6 record
            log of 6 to 1 odds:  log(6/1) = log(6)     = 1.79
            log of 1 to 6 odds:  log(1/6) = log (0.17) = -1.79
              - symmetrical results
              - with log function, the distance from the origin (or 0) is the same for 1 to 6 or 6 to 1 odds


  logit function / log(odds)
    - the log of the ratio of the probabilities is called the 'logit function' and forms the basis for logistic
      regression

      logit function = log [ p / (1 - p)]  = log(odds)   where p: probability

    - histogram of the log(odds) is the shape of a normal distribution (center: x = 0)
    - this makes log(odds) useful for solving certanin statistics problems - specifically one where we are
      trying to determine probabilities about win/lose or yes/no, or true/false types of situations


   in Summary:
    - odds are the ratio of something happening (i.e. team A winning) to something not happening
      (i.e. team A not winning)
    - the log(odds) is just the log of the odds
    - log(odds) makes things symmetrical, easier to interpret and easier for fancy statistics

--------------------------------------------------
Odds Ratios and Log(Odds Ratios), Clearly Explained!!!
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=8nm0G-1uJzA

  NOTE: log is actual log_e = ln

  Summary
    - Odds Ratios and Log(Odds Ratios) are like R-Squared - they describe a relationship between two things. 
    - And just like R-Squared, you need to determine if this relationship is statistically significant. 
    - This StatQuest goes over all these details so that you are ready for any odds ratio and log(odds ratio) 
      situation that you might get yourself into!!!


  odds ratio
    - refers to "ratio of odds", i.e.:    odds 1 / odds 2

       odds ratio between odds 1: 2 to 4 and  odds 2: 3 to 1  
             = [2/4]  / [3/1] = 0.1667
       odds ratio between odds 1: 3 to 1 and  odds 2: 2 to 4  
             = [3/1]  / [2/4] = 6

     - odds ratio are unsymmetrical
        - if the denominator is larget that the numerator, then the odds ratio will go from 0 to 1
        - if the numerator is larget that the demoninator, then the odds ratio will go from 1 to infinity
     - log_e can be used to make them symmetrical: log(odds ratio)

       log of odds ratio between odds 1: 2 to 4 and  odds 2: 3 to 1  
             = log([2/4]  / [3/1]) =  log_e(0.1667) = -1.79
       log of odds ratio between odds 1: 3 to 1 and  odds 2: 2 to 4  
             = log([3/1]  / [2/4]) =  log_e(6)      = 1.79


   use odds ratio to determine relationship
     - the oddss ration and log(odds ratio) are like R-squared; they indicate a relationship between
       two things (in this case, a relationship between mutated gene and cancer)
       - just like R-squared, larger values mean the gene is a good predictor of cancer
       - smaller values mean that the mutated gene is not a good predictor of cancer
       - However, just like R-squared, we need to know if this relationship is statistically signficant

      example: determine if there is a relationship between the mutated gene and cancer

                   Has Cancer
                    Yes  No                                                                     odds ratio:
                   ----------        if they have the mutated gene, the odds they have cancer:   23/117
       has      Yes| 23  117                                                                    --------- = 6.88
       mutated  No |  6  210         if they do not have mutated gen, the odds they have cancer: 6 / 210
       gene                            

       -> that is have the mutated gene, they odds they have cancer is 6.88 greater than if they don't
          have the mutated gene

       -> log_e(odds ratio) = log_e(6.88) = 1.93

    3 ways to determine if odds ratio (or log(odds ratio) statistically significant 
      1. Fisher's EXact Test
      2. Chi-Square Test
      3. Wald Test
      - no consensus on which method is best
      - sometimes thye mix and match above methods for the p-value and confidence interval calculations


    Fisher's Exact Test
      step 1:
      - what StatQuest on "Enrichment Analysis using Fisher's Exact Test and the Hypergeometric Distributions"
      Next:
        calcuated p-value: 0.00001

    Chi-Square Test
      - compares the observed values to the expected values that assume there is no relationship between
        the mutated gene and cancer

          Calcuated the probability of having cancer
              = 23 + 6 / total [23 + 6 + 117 + 210] = 29/356 = 0.08

              therefore, based on the probability of having cancer, people with cancer is expected to be:
                     with mutated gene:      0.08 x (23 + 114) = 11.2
                     wihout mutated gene     0.08 x (6 + 210)  = 17.3 

            Expected Values:
                           Has Cancer
                           Yes     No         
                          -------------        
              has      Yes| 11.2  128.8      
              mutated  No | 17.3  198.7      
              gene                            

                do Chi-Square test to compare the observed and expected values:
                  and the p-value is:
                      0.00001 with continuity correction
                      0.000004 without continuity correction
                      -> see statquest on Chi-square test

    Wald Test
      - commonly used to determine the significants of odds-ration in logistic regression and to 
        calcuate the confidence intervals
      - takes advantage of the fact the log(odds ratios), just like log(odds), are normally distributed
      - wald test looks to see how many std deviations the observed log(odds ration) is from 0
      for calculate std deviation using random numbers / histogram:
        - when there is no  difference in the odds, the log(odds ratio) for large number of random values equals 0
        - gives example of using historgram and plotting log(odd ration) for large number of random values
        -> this method calculate std deviation: 0.43

      estimated std deviation from observed values:
            √ (∑ [1/observed values]) = 
            √([1/23] + [1/117] + [1/6]  + [1/210])   = 0.47

      find the number of std deviations away from the mean of the distribution:

             log(odds ratio) / estimated std deviation =  log_e(6.88) / 0.47 = 1.93 / 0.47 = 4.11

       General rule of thumb with normal distributions
         - anything further that 2 standard deviations from the mean will have a p-value < 0.05,
           so we know our log(odds ration) is statistically significant

       get a precise 2-sided p-value:
         - add up areas under the curve for points > 1.93 and for points < -1.93
         - however, this traditionally done using a std normal curve (mean = 0, and std deviation = 1)
         - that meanas adding up the area under the curve for points that are > 4.11 and for 
           pointsa < -4.11 where 4.11 is the number of std deviations that log(odds ratio) is 
           away from the mean
         - ultimately, the p-value that the mutated gene does not have a relationship with cancer is: 0.00005



--------------------------------------------------
StatQuest: Logistic Regression
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=yIYKR4sgzI8

  Summary: 
    - Logistic regression is a traditional statistics technique that is also very popular as a machine learning tool. 
    - In this StatQuest,  I go over the main ideas so that you can understand what it is and how it is used.

  Linear Regression
    - fit a line to your data 2D data (i.e. x-axis: weight, y-axis: size)
    Normal regression (2D) - could do a lot of things:
     1. calculate R**2 and determine if weight and size are correlated. Large values imply a large effect
     2. Calcuate a p-value to determine if R**2 value is statistically significant
     3. use the line to predict size for a given weight
    Multiple Regressions (3D+)
    - 3D i.e.: predict size using weight and blood volume
    - alternatively, we could say we are trying to model size using weight and blood volume
    do the same things a normal regression did:
     1. calculate R**2
     2. calculage the p-value
     3. Predict size give and weight and blood volume

     Compare simple to compilicated linear regressions (i.e normal [2D] vs multiple [3D+])
       - tells us if we need to measure additional features (i.e. weight vs size and blood volume) to
         accurately predict size or if we can just use less features (i.e. weight)

  Logistic Regression
    - predicts whether something is 'true' or 'false' instead of predicting something continuous
      like size
    - instead of fitting a line to the data, logistic regression fits an "S" shaped "logistic funtion"
    - the "S" curve goes from '0' to '1'
    - the curve tells you the probability of 'true' or 'false'
    - although logistic regression tels the probability that something is 'true' or 'false', (i.e. obsese,
      not obsese), it is usually used for classification
    -  works with both continuous data (i.e. weight, age) and discrete data (i.e. genotype, astrological sign)
    - can test if each variable effect on the prediction is significantly different from zero -
      that is, does the variable help with the prediction; use 'Wald's Test to figure this out
    - unlike normal [linear] regression, we cannot easily compare complicate [using more features] to simple model 
    - uses maximum likelihood (instead of calculating Residuals and R-squared as with linear regressions)
      - see Statquest on Maximum Likelihood
      - you pick a probability, scaled by feature(s) (i.e. weight) of observing if target is true/false
        (i.e. obsese/not obsese), and then  calculate the likelihood of observing the 'true' case (i.e
        obsese) and the 'false' case (i.e. not obsese)
        - then shift the line and calculate a new likelihood of the data
        - repeat shift the line and calculating a new likelihood
        - finally, the curve with the maximum likelihood is selected


  in Summary:
    - Logistic Regression can be used to classify samples
    - it can use different types of data including continuous (i.e. weight) and discrete (i.e. genotype)
       to do classification
    - can assess which variables are useful for classifying samples (if not useful, it is 'totes useless')
     
--------------------------------------------------
Logistic Regression Details Pt1: Coefficients
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=vN5cNN2-HWE

  Summary:
   - When you do logistic regression you have to make sense of the coefficients. 
   - These are based on the log(odds) and log(odds ratio), but, to be honest, the easiest way to 
     make sense of these are through examples. 
   - In this StatQuest, I walk you though two Logistic Regression Examples, step-by-step, and show 
     you exactly how the coefficients are derived and how to interpret them.


  Logistic Regression Overview
    - classify by using a probability between 0 (true) and 1 (false)
    - instead of fitting a line to the data, logistic regression fits an "S" shaped "logistic funtion"
    - the "S" curve goes from '0' to '1'

   Generalized Linear Models (GLMs)
    - Logistic Regression is a specific type of Generalize Linear Model (GLM)
    - GLMs are a generalization of the concepts and abilities of regular Linear Mode


  Logistic Regressions using continuous variable (i.e. weight)
    - this type of Logistic regression is closely related to "Linear Regression" (a type
      of linear model)
    - with logistic regression, the y-axis is confined ot probability values between 0 and 1
    - to solve the probability, the y-axis is transformed from the probability of 'true' (i.e. obsesity)
      the log_e(odds of obseity) which can go from -infinity to +infinity 

  obseity example 
    - transform y-axis from a "probility of obesity" scale to a "log(odds of obseity)" scale
      where log_e(odds of obseity) = log_e( p / [1 - p])  where 'p' is obsese probabity  <- using logit function
    - calcuate for

          y-axis mid point: 0.5 -> log_e(0.5/[1-0.5]) = log_e(1) = 0
             -> center of new y-axis
          other y-axis point conversions:
                            0.731  = log_e (0.731/[1-0.731]) = log_e(2.717) = 1 
                            0.88   = log_e (0.88/[1-0.88])   = log_e(7.33)  = 2 
                            0.95   = log_e (0.95/[1-0.95])   = log_e(19)    = 3 
                            1      = log_e (1/0) = log_e(1) - log_e(0) = log_e(1) - (-infinity) = +infinity 
                                 -> for original samples labeled obsese
          New y-axis:
             - log_e(odds of obesity) goes from: -infinity <-> 0 <-> +infinity
             - transforms "S" curved line to a straight line
             - even though the graph with the "S" curved line is what we associate with
               logistic regression, the coefficients are presented in terms of the log_e(odds) graph
             - See StatQuest "Logistic Regression Details Part 2: Fitting a Line with Maximum Likelihood"
               for details on how this line fits the data
             - the coefficient for the line are what you get when you do the logistic regression

                     y = -3.48 + 1.83 x weight

                               Estimate Std. Error z value  Pr(<|z|)
                    (Intercept)  -3.476      2.364  -1.471    0.1414
                    weight        1.825      1.088   1.678    0.09346

              - intercept z value is the estimated intercept divided by the std error
                         -3.476 / 2.364 = -1.471
                 - in other words, the z-value is the number of std deviations the estimated intercept
                   is away for 0 a std normal curve
                   - this the Wald's Test talked about in the "odds ratio" StatQuest
                   - since the estimate is less than 2 std deviations away from 0, we know it is not
                     statistically significant
                   - this is confirmed by the large p-value; the area under a std normal curve that
                     is further than 1.471 (and -1.471) std deviations away for 0 

              - slope z value is the estimated slope divided by the std error
                          1.825 / 1.088 =  1.678



  Logistic Regressions using discrete variable (i.e. mutated gene)
    - very similar to how a t-test is done using linear model

   T-test for linear models review 
     example: whether obese mice based on mutated gene mice 
     - measure size of mice with normal gene and size of mice with mutated gene
     - fit two lines to the data:
        - 1st line represents the mean size of the mice with normal gene 
        - 2nd line represents the mean size of the mice with mutated gene 
     - calcuate:

        size = mean_normal X B_1 + (mean_mutant - mean_normal) x B_2

        - pair this equation with a design matrix to predict the size of a moust give that is has
          the normal or mutated version of the gene
      
          Design matrix
          B_1    B_2
           1      0     - 1st column turns corresponds to values for B_1, and it 'turns on" the
           1      0       the first coefficient, mean_normal
           1      0
           1      0     - 1st column turns corresponds to values for B_2, and it 'turns on" the
           1      1       the 2nd coefficient, (mean_mutant - mean_normal), "off" or "on" depending on
           1      1       whether it is a '0' or a '1'
           1      1
           1      1     - when we do a 't-test' this way, we are testing to see if this coefficient
                          (mean_mutant - mean_normal), is equal to 0


  Logistic Regressions using discrete variable (i.e. mutated gene) steps:
    - transform the y-axix from the probability of being 'true' (obese) to log(odds of obesity)
     - fit two lines to the data:
        - 1st line represents the log(odds of obesity) for mice with the normal gene.
          let's call this the log(odds gene_normal)
           log(2/9) = log(0.22) = -1.5
        - 2nd line represents the log(odds of obesity) for mice with the mutated gene.
          let's call this the log(odds gene_mutated)
           log(7/3) = log(2.33) =  0.85
      - these two lines come together to for the coefficients in this equation                    

          size = log(odds gene_normal) X B_1 + (log(odds gene_mutated) - log(odds gene_normal)) x B_2

        convert: log(A) - log(B)  to: log(A/B):

          size = log(odds gene_normal) X B_1 + log(odds gene_mutated / odds gene_normal) x B_2

       - tells us, on a log scale, how much having a mutated gene increases (or decreases) the
         odds or a mouse being obese

       - calculate

          size = log(2/9) X B_1 + log([7/3] / [2/9]) x B_2
               = -1.5 x B_1     +  2.35 x B_2

        - and these are what you get when you do logistic regressions

                               Estimate Std. Error z value  Pr(<|z|)
                    (Intercept)  -1.5041     0.7817 -1.924    0.0544
                    weight        2.3514     1.0427  2.255    0.0241 

            - the intercept is the log(odds gene_normal)
            - the "geneMutual" term (slope) is the log(odds ratio) that tells you, on a log scale,
              how much having the mutated gene increases or decreases the odds of being obese

            - the z-values (aka the Wald's Test values) that tell you how many std deviations the
              estimated coefficients are away from 0 on a std normal curve

            - the intercept z-value, -1.9, tells us that the estimated value for the intercept -1.5,
              is less than 2 std deviations from 0, and thus not significantly different from 0
              - confirmed by a p-value greater than 0.05 (p-value: 0.0544)

            - the z-value for 'geneMutant', log(odds ratio) that describes the odds of being obese,
              is greater than 2 (z-value: 2.255), suggesting it is statistically signficant
              - confirmed by a p-value less than 0.05 (p-value: 0.0241)

   in Summary:
      - in terms of the coefficients, logistic regressions is the exact same as linear model except
        the coefficients are in terms of log_e(odds)
      - this means the things we can do with linear models, like multiple regressions and ANOVA,
        can be done using logistic regressions. All we have to remember is the the scale for the
        coefficients is log(odds)

--------------------------------------------------
Logistic Regressions Details Part 2: Maximum Likelihood
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=BfKanl1aSG0


  Summary
    - This video follows from where we left off in Part 1 in this series on the details of Logistic Regression. 
    - This time we're going to talk about how the squiggly ['S' curved] line is optimized to best fit the data.

  Logistic regression example: 
     - determine the effect of weight on obesity
     - goal, draw the best fitting "S" curve to fit the data
     - in logistic regression, the y-axis is transformed from probability of obesity (0 to 1), to the 
       log(odds of obesity) from (-infinity <-> 0 <-> +infinity)
        - this means the residuals (the distance from the data points to the line) are also equal to
          positive or negative infinity which means you cannot use least-squares to find the best fitting line
     - instead, you use Maximum Likelihood
       - first, project the original data points onto the candidate line using its log(odds) value
     - transform the candidates log(odds) to candidate probabilities using the formula:
          p = e**log(odds) / [ 1 + e**log(odds)]   where p: probability

          - which is just a reordering of the transfromation from probability to log(odds)

             log_e(p /[1 - p]) = log_e(odds)

             Converting:
                from: 
                      log_e(p /[1 - p]) = log_e(odds)
                to: 
                      p = e**log(odds) / [ 1 + e**log(odds)]   where p: probability
               
                 exponentiate both sides:
                     p / [1 - p]  = e**log(odds)

                 multiple both sides by (1 - p):
                     p  = (1 - p) e**log(odds) = e**logs(odds) - p x e**log(odds)

                 add pe**log(odds) to both sides
                     p + p x e**log(odds) = p[1 + e**log(odds)] = e**logs(odds) 
                               
                 divide both sides by (1 + e**log(odds)):
                     p = e**log(odds) / [1 + e**logs(odds)] 

         - example: take projected log(odds) data on the candidate line and transform to its probability
            p = e**log(odds) / [ 1 + e**log(odds)]   where p: probability
               substitute -2.1 for log(odds)
            p = e**-2.1 / [ 1 + e**-2.1] =  0.1  -> this is y-cordinate on the "S" curve   
         - repeat for all the log(odds) values 

         - use the observed status (obese or not obese) to calculate their likelihood give the shape 
           of the "S" Curved line

         - start by calculating the likelihood of the obese mice give the shape of the "S" curved line
           (projection on the the y-axis)
             - in other words, the likelihood that this mouse is obese, given the shape of the "S" curve
               is the same as the predicted probability
             - in this case, the probability is not calculated as the area under the curve, but
               instead is the y-axis value, and that's why it is the same as the likelihood

               likelihood of the obese mice were: 0.49, 0.91, 0.91, and 0.92

         - the likelihood for all of the obese mice is the product of the individual likelihood:
                0.49 x 0.91 x 0.91 x 0.92


         - for not obese mice: determine probability of not being obese which is:
              likelihood = (1 - probability the mouse is obese)

               likelihood of the not obese mice were: (1 - 0.9) (1 - 0.3), (1 - 0.1), and (1 - 0.1)

         - include the include the individual likelihoods for the "not obese" mice to the overall likelihood

           overall likelihood of the data = 
                0.49 x 0.91 x 0.91 x 0.92   x  (1 - 0.9) x (1 - 0.3) x (1 - 0.1) x (1 - 0.1)

            Note: Although it is possible to calculate the likelihood as the product of the 
            individual likelihoods, statisticians prefer to calculate the log of the likelihood instead

            log(likelihood of the given "S" curve) = 
                log(0.49) + log(0.91) + log(0.91) + log(0.92) + log(1 - 0.9) + log(1 - 0.3) + log(1 - 0.1) + log(1 - 0.1)
                = -3.77
                - with log of the likelihood or the "log-likelihood", we add the logs of individual 
                  likelihoods instead of multiplying 

            - this means that the log-likelihood of the original line is -3.77

            - now rotate the line and calculate its log-likelihood projecting the data onto it
            - transform log(odds) to probabilities
                  p = e**log(odds) / [ 1 + e**log(odds)]   where p: probability
            - and then calculating the log-likelihood
               log-likelihood final value = -4.15, this one is not a good as the first one
            - repeat: keep rotating the log(odds) line and projecting data onto it, and transforming 
              it to probabilities and calculating log-likelihood
                 
            Note: the algorithm that finds the line with the maximum likelihood is pretty smart - each
                  time it rotates the line, it does so in a way to increase the log-likelihood. Thus,
                  the algorithm can find the optimal fit after a few rotations

   There is more to Logistic Regression than fitting a line:
     - want to know if that line represents a useful model, and that means we need a R-squared value and p-value
     - but need to due it without the usual residuals -> in next statQuest lesson


--------------------------------------------------
Logistic Regression Details Pt 3: R-squared and p-value
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=xxFYro8QuXA

  Summary:
    - This video follows from where we left off in Part 2 in this series on the details of Logistic Regression.  
    - Last time we saw how to fit a squiggly line to the data. 
    - This time we'll learn how to evaluate if that squiggly ["S' Curved] line is worth anything. 
    - In short, we'll calculate the R-squared value and it's associated p-value.


  Calculate R-squared for Logistic Regression
    - no consensus on how to calcuate R-squared, and more than 10 different ways to do it
    - focus on a common method called "McFadden's Psuedo R-squared"

   How to calculate R-squared and related p-values for Linear Regression
     - R-squared is calculated using the Residual
     - Square the Residuals and sum them up
     - SS(fit) for "sum of the squares for the residuals" around the best fitting line
     - compare to SS(mean) "Sum of the squared residuals around worst fitting line. the 
       mean of the y-axis value

     - R-squared is the percentage of the variation around the mean that goes away when you fit a 
       line to the data
         - R-squared goes from 0 to 1
           - R-squared = 0: if SS(fit) = SS(mean)    -> variable provides no improvement
           - if data fits the line perfectly, SS(fit) = 0, then R-squared = 1

            R-squared =  [SS(mean) - SS(fit)] / SS(mean)

  Logistic Regression
     - residuals are all +infinity or -infinity, so we can not use them
     - instead, we can project the data onto the best fitting line, and then translate the log(odds)
       back to probabilities, and then calculate the log-likelihood of the data given the best 
       fitting "S" curved line
            - transform log(odds) to probabilities
                  p = e**log(odds) / [ 1 + e**log(odds)]   where p: probability
            log(likelihood of the given "S" curve) = 
                ∑ [log(true datapoints on "S" line)]  + ∑ [log(1 - false datapoints on "S" line)]  
                = previous video example:
                log(0.49) + log(0.91) + log(0.91) + log(0.92) + log(1 - 0.9) + log(1 - 0.3) + log(1 - 0.1) + log(1 - 0.1)
                = -3.77
                That is, log-likelihood of the fit line, LL(fit) = -3.77, and use it as a substitute for SS(fit)

     - now we need a measure of a poorly fitted line that analogous to SS(mean)
        - do this by calculating the log(odds of obesity) without takeing weight into account
        - the overall log(odds of obesity) is the just the log of the total number of obese mice (5) 
          divided by the total number of "not obese" mice (4)
              log(5/4) = 0.22

        - project the data on to the horizontal line at 0.22
            - transform log(odds) to probabilities
                  p = e**log(odds) / [ 1 + e**log(odds)]   where p: probability

                  p = e**0.22 / [ 1 + e**0.22] = 0.56

                  - this gives a horizontal line at p=056 on the probability normally "S" curved line

           Note: the overall log(odds), 0.22 translates to the overall probability of being obese, 0.56
           - in other words, we can arrive at the same solution by calculating the overall probability of obesity

             p = [number of obese mice] / [total number of mice] = 5 / 9 = 0.56

        - now calculate the log-likelihood of the data given the overall probability of obesity
                ∑ [log(true datapoints on "S" line)]  + ∑ [log(1 - false datapoints on "S" line)]  
                = previous video example:
                log(0.55) + log(0.55) + log(0.55) + log(0.55) + log(0.55) + 
                    + log(1 - 0.55) + log(1 - 0.55) + log(1 - 0.55) + log(1 - 0.55)
                = -6.18
           - call this the LL(overall probability) [bad fit measure] and use it as a subsititute for SS(mean)
                LL(overall probability) = -6.18

                Puedo R-squared = [LL(overall probability) - LL(fit)] / LL(overall probability)
                                = [-6.18 - -3.77] / -6.18 = 039

  Log-likelihood values for logistic regressions
    - when the model is a poor fit, the log-likelihood is a relative large negative number (i.e. -6.18)
    - when the model is a good fit, the log-likelihood is a value close to zero 
    - always be between 0 and -infinity because we are taking logs of values between 0 and 1
      (from probability curve)
     

   p-value calculation:
      2(LL(fit) - LL(overall probability) = a chi-squared value with the degrees of freedom equal to the
                                            difference in the number of parameters in the two models

        - LL(fit) has 2 parameters since it needs estimates for a y-axis intercept and a slope
        - LL(overall probability) has 1 parameters since it only needs estimates for a y-axis intercept

               Degrees of freedom = 2 - 1 = 1
                
        - LL(fit) has 2 parameters since it needs estimates for a y-axis intercept and a slope


--------------------------------------------------
Regularization Part 1: Ridge (L2) Regression Clearly Explained
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=Q81RR3yKn30

  Summary:
    - Ridge Regression is a neat little way to ensure you don't overfit your training data - essentially, you 
      are desensitizing your model to the training data. 
    - It can also help you solve unsolvable equations, and if that isn't bad to the bone,


  Simple Example: 
    - predict mouse size based on mouse weight
    - use linear Regression (AKA least squares) since data looks somewhat linear
    - fit a line to data that results in the miniumum sum of the squared residuals
      - determine equation 1: size = 0.9 + 0.8 x weight
    - case 2: new line using just 2 of the original data points , so the new line that overlaps both data points, 
      and therefore it minimum sum of squared residuals = 0
      -  equation 2:  size = 0.4 + 1.3 x weight
      -  uses just 2 original data points for training data, and the remaining 6 data points for testing data
      - this result in:
        - testing data having high variance (large sum of the squared residuals) result overfitting 

   Ridge Regression
     - find a new line that doesn't fit the training data as well
     - in other words, we introduce a small amount of Bias into how the new line is fit to the data
       - increasing bias to reduce the variance

   Linear Regression
     - when least squares determines values for the parameters using this equation:
        size = y-axis intercept + slope x Weight
        - it minimizes the "sum of the R-squared"

   Linear Regression + Ridge Regression
     - when least squares determines values for the parameters using this equation:
        size = y-axis intercept + slope x Weight
        - it minimizes the "sum of the R-squared"  PLUS   λ x slope**2
           slope**2 adds a penalty and lambda, λ, determines the weight of the penalty

    Case 2 and 1 with Ridge Regression

      -  equation 2:  size = 0.4 + 1.3 x weight  +  λ x slope**2    where λ = 1
                         penalty:  λ x slope**2 =  1 x 1.3**2 = 1.69
                      size = 0.4 + 1.3 x weight  +  1.69
                      -> reduces the slope by adding bias which then should reduce variance

      -  equation 2:  size = 0.9 + 0.8 x weight  +  λ x slope**2    where λ = 1
                         penalty:  λ x slope**2 =  1 x 0.8**2 = 0.64
                      size = 0.9 + 0.8 x weight  +  0.64


  Ridge Regression impact on Linear Regression Line
     - when the slope of the line is steep, then the prediction for size is very sensitive to 
       relatively small changes in weight
     - when the slope of the line is small, then the prediction for size is much less sensitive 
       to changes in weight 
     - ridge regression penalty results in a line that has a smaller slope 
       - this means predictions made with Regression penalty are less sensitive to changes in
         weight than with a least squares line

  Lambda, λ
     - λ can be any value from 0 to +infinity
     - when λ = 0, there is no ridge regression penalty
     - as λ increases, the Ridge Regression line slope gets smaller
     
   Determining Lambda value:
     - use Cross Validation, typically with 10-fold Cross Validation, to determin which one 
       results in the lowest variance
   

   Ridge Regression with Discrete variables
     - Ridge Regression also works with discrete variables line Normal Diet vs High Fat Diet to predict size

   Discrete example:  Normal Diet vs High Fat Diet to predict size

      Least Squares fit equation:  size = 1.5 + 0.7 x High Fat Diet
         where: 
            1.5: the equivalent of the y-intercept, corresponds to the average size of the mice on
                 the normal diet
            0.7: the equivalent of the slope, corresponds to the difference between the average size 
                 of the mice on the normal diet compared to teh mice on the high fat diet
                 - this will be called the "diet difference"
            High Fat Diet: 0 for mice on normal diet, or 1 for mice on the High Fat Diet

       Residuals
         - on normal diet, the residuals are the distance between the mice size and the Normal Diet mean
         - on high fat diet, the residuals are the distance between the mice size and the high fat Diet mean

       Ridge Regression penalty
         - λ x 'Diet Difference'**2 
         - as lambda, λ, gets larger, our prediction on the size of the mice on the high fat diet 
           becomes less sensitive to the difference between the 'Normal Diet' and the 'High Fat Diet'


   Ridge Regression applied to Logistic Regression
     - when applied to Logistic Regression, Ridge Regression optimizes the 'the sum of the likelihoods'
       instead of the 'squared residuals' because Logistic regression is solved using 'Maximum Likelihood'
    
      example: using 'weight' to predict if a mouse is 'obese' or 'not'
        equation:  obese = y-intercetp + slope x weight
          - ridge regression would shrink the estimate for the slope, making our prediction about 
            whether or not a mouse is obese less sensitive to 'weight'

              the sum of the likelihoods  +  λ x slope**2

  Complicated Model with Ridge Regression
    - in general, the Ridge Regression Penalty contains all of the parameters except for the y-intercept

    example: combine weight measure measurement data from 1st example with two Diets from 2nd example

      combined equation:  y-intercept + slope x Weight + diet difference x High Fat Diet

      Ridge Regression Penalty:
                  λ x (slope**2 + 'Diet Difference'**2) 


   Datasets with large number of parameters
     - normally, you need at least the number datapoint equal or greater than the number of parameters
     - it turns out that by adding the 'Ridge Regression Penalty', you can solve for all 10000 parameters
       with only 500 (or even fewer) samples (data points)
       - with Ridge Regression penalty, use:
                  λ x (slope1**2 + slope2**2  + slope3**2 + ... + slope 10000**2) 
      - more details in future statquest


   in Summary:
     - when the sample size are relatively small, the 'Ridge Regression' can improve predictions made from
       new data (i.e. reduce variance) by making the predictions less sensitive to the 'Training Data'
     - made less sensitive by adding the Ridge Regression Penalty:
             λ x slope**2
     - the 'Ridge Regression Penalty is  λ times the sum of all squared parameters, except for the
       y-intercept
     - the lambda, λ, value is determined using Cross Validation
     - even when there is not enough data to find the Least Squares parameter estimates, Ridge Regression can
       still find a solution with 'Cross Validation' and the 'Ridge Regression Penalty'


--------------------------------------------------
Regularization Part 2: Lasso (L1) Regression Clearly Explained
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=NGf0voTMlcs

  Summary:
    - Lasso Regression is super similar to Ridge Regression, but there is one big, huge difference between the two. 
    - In this video, I start by talking about all of the similarities, and then show you the cool thing that 
      Lasso Regression can do that Ridge Regression can't.


  Ridge Regression
    - increase the bias which also reduces the variance by adding 'Ridge Regression Penalty':
             penalty: λ x slope**2
    - by adding a little bias which slightly reducing the fit to the training data, it provided a better 
      long term predictions
    - Linear regression plus Ridge Regression Penalty equation:
           'sum of the squared residuals' + λ x slope**2


  Lasso Regression
    - increase the bias which also reduces the variance by adding 'Lasso Regression Penalty':
             penalty: λ x |slope|
    - Just like Ridge Regression, λ can be any value  from 0 to +infinity and value is determined
      using Cross Validation
    - Linear regression plus Lasso Regression Penalty equation:
           'sum of the squared residuals' + λ x |slope|

  Complicated Model with Lasso Regression
    - in general, the Lasso Regression Penalty contains all of the parameters except for the y-intercept
    example: combine weight measure measurement data from Part 1 1st example with Part 1 two Diets from 2nd example

      combined equation:  y-intercept + slope x Weight + diet difference x High Fat Diet

      Lasso Regression Penalty:
        - includes all of the estimated parameters except for the y-intercept

                  λ x (|slope| + |'Diet Difference'|) 

   Difference between Lasso Regression and Ridge Regression
     - Ridge Regression can only shrink the slope asymptotically close to zero while Lasso Regression 
       can shrink the slop all the way to zero
     asymptote
       - a line that continually approaches a given curve but does not meet it at any finite distance.
     - With Lasso, as λ is increased, the useful parameters will shrink a little bit, and the useless 
       parameters will go all the way to zero
     - Since Lasso Regression can exclude useless variables from equations, it is a litter better than
       Ridge Regression at reducing the Variance in models that contain a lot of useless variables
     - in contrast, Ridge Regression tends to do a little better when most variables are useful


  In Summary:
    - Ridge Regression is very similar to ... --> 'sum of the squared residuals' + λ x slope**2

      To Lasso Regression                     --> 'sum of the squared residuals' + λ x |slope|

     - Big difference is that Lasso Regression can exclude useless variables from equations
        - this makes the final equation easier and simplier to interpret

--------------------------------------------------
XGBoost Part 1 (of 4): Regression
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=OtD8wVaFm6E

  Summary:
    - XGBoost is an extreme machine learning algorithm, and that means it's got lots of parts. 
    - In this video, we focus on the unique regression trees that XGBoost uses when applied to Regression problems.


  XGBoost 
    - ML algorithm with lots of parts:
      Gradient Boost-(ish), Regularization, A unique Regression Tree, Approximately Greedy Algorithm, 
      Weighted Quantile Sketch, Sparsity-Aware Split Findings, Parallel Learning, Cache-Aware Access, & 
      Blocks for Out-of-Core Computation
    - session assumes you are already familar with Gradient Boost & Regularization so it starts with 
      'A Unique Regression Tree'

  XGBoost
     - designed to be used with large, complicated datasets

  Simple training data:
     - drug dose (mg) impact on Drug Effectivenes 
     - 4 data points: 2 observation with positive effectiveness, and 2 with negative effectivenes 

           Drug Effectiveness
         10 -
            |     X     
          5 -   x
            |
          0 -  ------------ 0.5 initial prediction
            |
         -5 -
            |        x
        -10 -  x
            |
        -15 |----|----|-----------
            0    20   40

     step 1: Make initial prediction
        - can be anything, but by default it is 0.5, regardless of whether you are using XGBoost for
          Regression of Classification
        - residuals, the difference between the observed and Predict values, show us how good the initial
          prediction is
      step 2: Build first XGBoost tree
        - like Gradient Boost, XGBoost fits a Regression Tree to fit the residuals
        - But, XGBoost uses a unique Regression tree that video calls "XGBoost Tree"
        - there are many ways to build XGBoost tree - video focuses on the most common way for Regression
         tree steps:
          - build a single leaf with all the residuals:  -10.5, 6.5, 7.5, -7.5
            - calculate a quality or similarity score for the residuals of each leaf [node]
              similarity score = (Sum of the Residual)**2 / [number of Residuals + λ ]
                 λ: regularization parameter
                 calculate with λ = 0: (-10.5 + 6.5 + 7.5 + -7.5)**2 / [4 + 0] = (-4)**2/4 = 4
                 - when the residuals in a node are very different, they cancel each other out and
                   the Similarity Score is relatively small
                 - when the residuals are similar, or there is just one of them, they do not cancel out and
                   the Similarity Score is relatively large
            - calculate gain score between child nodes and parent node
               Gain = child_Left_similarity + child_Right_similarity - parent_similarity
            - repeat for each potential split. 
            - Use split with the highest gain score

          - see if splitting similar Residuals into 2 group does a better job clustering them
             - for each attempted split, calculate the similarity score for each leaf
               and then gain score between child nodes (leaves) and parent node (leaf)
             - select split with largest gain score

           2_level_1a. try split between 1 and 2 dosage levels to create 2 leaves using their average 
               dosage (10 + 20) / 2 = 15

                                      Dosage < 15           similarity: 4
                                     T/         \F
              similarity:          -10.5    6.5,7.5, -7.5   simiarity = (6.5 + 7.5 -7.5)**2/[3 + 0] = 14.08

                Gain = Left_similarity + Right_similarity - Root_similarity

                Gain for 1st leaves: 110.25 + 14.08 - 4 =  120.33

            2_level_1b. try split between 2 and 3 dosage levels to create 2 leaves using their average 
               dosage (20 + 25) / 2 = 22.5
                                          
                                        Dosage < 22.5         similarity: 4
                                       T/         \F
                similarity: 8    -10.5,6.5       7.5, -7.5    similarity: 0

                  Gain for 1st leaves:  8 + 0 - 4 =  4

               - Since the Gain for 'Dosage < 22.5' (gain = 4) is less than the Gain for 'Dosage < 15' (gain = 120.33),
                 'Dosage < 15' is better at splitting the Residuals into clusters of similar values


            2_level_1c. try split between 3 and 4 dosage levels to create 2 leaves using their average 
               dosage (25 + 35) / 2 = 30

                                           Dosage < 30           similarity: 4
                                          T/         \F
                 similarity: 4.08   -10.5,6.5,7.5     -7.5       similarity: 56.25

                 Gain for 1st leaves:  4.08 + 56.25 - 4 =  4 = 56.33

                - Since the Gain for 'Dosage < 30' (gain = 56.25) is less than the Gain for 'Dosage < 15' (gain = 120.33),
                  'Dosage < 15' is better at splitting the Residuals into clusters of similar values

            2_level_2. Use 1st level split for 'Dosage < 15' with the high gain, and now split 2-level 
              in 2nd level:

             2_level_2a:  try split between 1 and 2 dosage levels to create 2 leaves using their average 
               dosage (20 + 25) / 2 = 22.5

                                      Dosage < 15           
                                     T/         \F
                                   -10.5       Dosage < 22.5     simiarity:  14.08
                                               /        \
                  similarity: 42.25          6.5       7.5,-7.5   similarity: 0

                Gain for 1st leaves: 42.25  - 14.08 = 28.17 

              2_level_2b: try split between 2 and 3 dosage levels to create 2 leaves using their average 
                dosage (25 + 35) / 2 = 30

                                      Dosage < 15           
                                     T/         \F
                                   -10.5       Dosage < 30    simiarity: 14.08
                                               /        \
               similarity: 98          6.5,7.5         -7.5   similarity: (-7.5)**2/1 = 56.25 
               (6.5 + 7.5)**2 / 2 

                Gain for 1st leaves: 98 + 56.25 - 14.08 =  140.17

                - Since the Gain for 'Dosage < 30' (gain = 140.17) is less than the Gain for 'Dosage < 22.5' (gain = 28.17),
                  'Dosage < 30' is better at splitting the Residuals into clusters of similar values for 2nd level


                - to keep this example simple, a 3rd level will not be added to split 6.5 and 7.5
                - However, the default is to allow up to 6 levels

      step 3: Prune XGBoost tree based on its Gain values
         start by picking gamma, γ,  value (i.e. 130 or 150) 
         - calculate the difference between the 'Gain' associated with the lowest branch in the tree
           and the γ value
             - if 'Gain - γ < 0',  remove branch, else do not remove branch 
             - if branch, is not remove, then the branch'es parent node will not be remove even if 'Gain - γ < 0' 

      step 4: Repeat step 2 Building a XGBoost Tree but with Similarity scores calcuated with λ = 1

              similarity score = (Sum of the Residual)**2 / [number of Residuals + λ=1 ]
              root similarity score with λ=1 =  (-10.5 + 6.5 + 7.5 + -7.5)**2 / [4 + 1] = (-4)**2/5 = 3.2

           Note: λ is a regularization parameter, which means it is intended to reduce the prediction's
                 sensitivity to individual observations

                                      Dosage < 15           similarity: 3.2
                                     T/         \F
              similarity: 55.12    -10.5    6.5,7.5, -7.5   simiarity = (6.5 + 7.5 -7.5)**2/[3 + 1] = 10.56
              (-10.5)**2/[1 + 1]

                - when λ > 0, the similarity scores are smaller, and the decrease is 'inversely proportional
                  to the number 'Residuals' in the node

                Gain = Left_similarity + Right_similarity - Root_similarity

                Gain for 1st leaves: 55.12 + 10.56 - 3.2 =  62.48   (Note: Gain with λ = 0: 120.33)


                - when λ > 0, it is easier to prune leaves because the gain values are smaller

      step 5: Calculate Output values for leaves

         Output value = Sum of Residuals / [Number of Residuals + λ]

                                      Dosage < 15           
         output[λ=0]                 T/         \F
        -10.5/[1 + 0]=-10.5      -10.5       Dosage < 30        output: (6.5 + 7.5 + -7.5) / [3 + 0] = 2.17
                                               /        \
         output: (6.5 + 7.5) / 2 = 7     6.5,7.5         -7.5   output (-7.5)/ [1 +0] = -7.5

             output left leaf λ = 0: -10.5 /[1 + 0] = -10.5
                              λ = 1: -10.5 /[1 + 1] =  -5.25

             - when λ > 0, it reduces the amount that an individual observation adds to the overall preduction
             - default: λ = 0 (no regularization)


      step 6: Make predictions
        - just like Gradient Boost, XGBoost makes new predictions by starting with the initial Predictions
          and adding the output of the Tree scaled by the learning rate, eta (η) 
        - default learning rate η = 0.3 

        sample 1 predicted value (dosage = 10, effectiveness = -10):
            initial effectiveness + η x output = 0.5 + 0.3 x -10.5 = -2.65

            - the new prediction has taken a small step from initial prediction (0.5) 
              towards observed value (-10)

        sample 2 & 3 predicted value (dosage = 20, effectiveness = 5; and dosage = 25, effectivenes = 7?):
            initial effectiveness + η x output = 0.5 + 0.3 x 7 = 2.6

            - the new prediction has taken a small step from initial prediction (0.5) 
              towards observed values (5 & 7?)

        sample 4 predicted value (dosage = 30, effectiveness = -7 ?):
            initial effectiveness + η x output = 0.5 + 0.3 x -7.5 = -1.75

            - the new prediction has taken a small step from initial prediction (0.5) 
              towards observed value (-7 ?)


      step 7: Make another tree based on residuals from previous tree - repeat steps 2 - 6
          - new predictions uses output values from both trees
            initial effectiveness + η x (output_tree1 + output_tree2 + ... + output_treeN)

          - repeat this step until residuals are super small or reach maximum number of trees


  In Summary, when building XGBoost Trees for Regression
    - calculate similarity sCores
       similarity score = (Sum of the Residual)**2 / [number of Residuals + λ ]
         λ: regularization parameter
    - calculate 'Gain' to determine how to split the data
         Gain = Left_similarity + Right_similarity - Root_similarity
    - prune the tree by calculating the difference between the 'Gain' values and a user define
      'Tree Complexity Parameter, γ (gamma)
          Gain - γ => If negative, prune node, else do not prune 
       - if we prune, repeat for parent: 'Gain - γ' if negative prune node

     - calculate output values for the remaining leaves (nodes)
         Output value = Sum of Residuals / [Number of Residuals + λ]

     - make predictions
            initial effectiveness + η x (output_tree1 + output_tree2 + ... + output_treeN)

      -  λ is a  regularization parameter 
         - when λ > 0, it results in more pruning, by shrinking the 'Similarity scores', and it results
           in smaller 'output values for the leaves

--------------------------------------------------
XGBoost Part 2 (of 4): Classification - XGBoost Trees for Classification
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=8b1JEDvenQU

  Summary:
    - In this video we pick up where we left off in part 1 and cover how XGBoost trees are built for Classification.


  XGBoost 
    - ML algorithm with lots of parts:
      Gradient Boost-(ish), Regularization, A unique Regression Tree, Approximately Greedy Algorithm, 
      Weighted Quantile Sketch, Sparsity-Aware Split Findings, Parallel Learning, Cache-Aware Access, & 
      Blocks for Out-of-Core Computation
    - Part 1 provided an overview of how XGBoost Trees are built for Regression
    - In Part 2 (this lesson) will provide an overview of how XGBoost Trees are built for Classification


  Simple Training data 
    - consist of 4 different drug dosages
    - 2 green dots indicate drug was effective (dosages: 3? & 17?)
    - 2 red dots indicate drug was NOT effective (dosages: 8? & 11?)
         
                    1 -      x  x
                      |
       Probability    |
       that the   0.5 -  ------------ 0.5 initial prediction
       Drug is        |
       Effective      |         
                    0 -   x            x
                      |----|----|----|----|---
                      0        10        20


     step 1: Make initial prediction
        - can be anything, but by default it is 0.5, regardless of whether you are using XGBoost for
          Regression of Classification
        - 1 = drug is effective,  0 = drug is NOT effective, 
        - residuals, the difference between the observed and Predict values, show us how good the initial
          prediction is
      step 2: Build first XGBoost tree
        - fit a tree to the residuals
          - select node splitting based gain values that are calculated using similarity scores

             Similarity score = (∑ Residuals_i)**2  / (∑ [PreviousProbability_i x (1 - PreviousProbability_i)] + λ)

              - λ (lambda) is a regularization parameter that reduces the 'Similarity Score', which makes leaves
                easier to prune (default: λ = 0)

         initial (root) leaf:     [-0.5,0.5,0.5,-0.5] 
            similarity score = 0**2  / ... = 0

           2_level_1a. try split between 3 and 4 dosage levels to create 2 leaves using their average 
               dosage (10 + 20) / 2 = 15

                                          Dosage < 15           similarity: 0
                                         T/         \F
              similarity:0.33   -0.5,0.5,0.5        -0.5        simiarity: 1

            left leaf with λ = 0:
               Similarity score = (∑ Residuals_i)**2  / (∑ [PreviousProbability_i x (1 - PreviousProbability_i)] + λ)
                                = (-0.5 + 0.5 + 0.5)**2 / [(0.5 x (1-0.5)) + (0.5 x (1-0.5)) + (0.5 x (1-0.5)) + 0]
                                = 0.5**2  / 3 x (0.5 x 0.5)  = 0.33

            right leaf with λ = 0:
               Similarity score = (∑ Residuals_i)**2  / (∑ [PreviousProbability_i x (1 - PreviousProbability_i)] + λ)
                                = (0.5)**2 / [(0.5 x (1-0.5)) + 0]
                                = 0.5**2  / (0.5 x 0.5)  = 1

                Gain = Left_similarity + Right_similarity - Root_similarity

                Gain for 1st leaves: 0.33 + 1 - 0 = 1.33
                
           2_level_1b. try split between 2 and 3 dosage levels to create 2 leaves using their average 
               dosage (8? + 12? / 2 = 10
                   -> repeat step 2_level_1a
                   -> gain was less than 2_level_1a

           2_level_1c. try split between 1 and 2 dosage levels to create 2 leaves using their average 
               dosage (4? + 8? / 2 = 6
                   -> repeat step 2_level_1a
                   -> gain was less than 2_level_1a



           2_level_2. Use 1st level split for 'Dosage < 15' because it had the highest gain, and now split 2-level 
              in 2nd level:

               - just by examination, you can tell similarity score for 2nd level leaves between 
                 dosage 1 and 2 will be less than 2 and 3  since dosage 1 (-0.5) cancels dosage 2 (0.5),
                 so just split between dosage 2 and 3
                     
           2_level_2a:  split between 2 and 3 dosage levels to create 2 leaves using their average 
               dosage (4? + 8?) / 2 = 5


                                          Dosage < 15          similarity: 0
                                         T/         \F
              similarity: 0.33     Dosage < 5      -0.5        simiarity: 1 
                                 /       \          
              similarity: 1  -0.5       0.5,0.5   similarity: 2

            left leaf with λ = 0:
               Similarity score = (∑ Residuals_i)**2  / (∑ [PreviousProbability_i x (1 - PreviousProbability_i)] + λ)
                                = (-0.5)**2 / [(0.5 x (1-0.5)) + 0]
                                = 0.5**2  / (0.5 x 0.5)  = 1

            right leaf with λ = 0:
               Similarity score = (∑ Residuals_i)**2  / (∑ [PreviousProbability_i x (1 - PreviousProbability_i)] + λ)
                                = (0.5 + 0.5)**2 / [(0.5 x (1-0.5)) + (0.5 x (1-0.5)) + 0]
                                = 1**2  / 2 x (0.5 x 0.5)  = 2

                Gain = Left_similarity + Right_similarity - Root_similarity
                     = 1 + 2 - 0.33 = 2.66

                     - since 2.66 > 0.66, we will use 'dosage > 5' as the threshold fo this branch
                        where 0.66 is gain score for the split between dosage 1 and 2

                    - limiting tree levels to 2 for demo
                    -  however, XGBoost also has a threshold for the minimum number of 'Residuals'
                       in each leaf

                mininum number of Residuals in each leaf
                  - determined by calculating the 'Cover' (denominator of similar score equation without λ)
                  - cover equation for classification:  
                    cover = ∑ [PreviousProbability_i x (1 - PreviousProbability_i)]
                  - cover equation for regression:  
                        cover = number of Residuals
                  - by default, the cover minimum value is 1
                    - thus, by default, XGBoost for Regression can have as few as 1 Residual per leaf

                    Cover classification example
                                   Dosage < 5    
                                 /       \          
                             -0.5       0.5,0.5 

                       left leaf cover:  0.5 x (1 - 0.5) = 0.25
                       right leaf cover: 0.5 x (1 - 0.5) +  0.5 x (1 - 0.5) =  2 x (0.5 x 0.5) = 0.5
                       - with default cover, XGBoost would not allow these leaves

                       - in order to prevent this from being the worst example ever, let's set the minimum
                         cover = 0
                       - this means setting the 'min_child_weight' parameter to '0'
                      

      step 3: Prune XGBoost tree based on its Gain values
         start by picking gamma, γ,  value (i.e. 2 or 3 for this example ) 
         - calculate gain score between child nodes and parent node
               Gain = child_Left_similarity + child_Right_similarity - parent_similarity
         - calculate the difference between the 'Gain' associated with the lowest branch in the tree
           and the γ value
             - if 'Gain - γ < 0',  remove branch, else do not remove branch 
             - if branch, is not remove, then the branch'es parent node will not be remove even if 'Gain - γ < 0' 

           - remember that λ (lambda), the regularization parameter, reduces the Similarity scores, and
             lower similarity scores result in lower gain
           - in other words, the values for λ greater than 0 reduce the sensitivity of the tree to 
             individual observations by pruning and combining them with other observations

      step 4: Calculate the output values for the leaves
         - Classification Output values for a leaf:

               output value = (∑ Residuals_i) / (∑ [PreviousProbability_i x (1 - PreviousProbability_i)] + λ)

            - note: with the exception of λ (lambda), regularization parameter, this is the same formula we
              used for unextreme Gradient Boost
            - When λ > 0, it reduces the amount that an observation adds to the new prediction [since it 
              reduces the output value]
              - that is, λ reduces the sensitivity to isolated observations


                                          Dosage < 15                           
                                         T/         \F
                                   Dosage < 5      -0.5                         
                                 /       \           -2 :output 
                              0.5       0.5,0.5                    
                       output: -2          2: output


            left bottom leaf output value with λ = 0:
               output value = (∑ Residuals_i)  / (∑ [PreviousProbability_i x (1 - PreviousProbability_i)] + λ)
                                = (-0.5) / [(0.5 x (1-0.5)) + 0]
                                = -0.5  / (0.5 x 0.5)  = -2

            right bottom leaf output value with λ = 0:
               output value = (∑ Residuals_i)  / (∑ [PreviousProbability_i x (1 - PreviousProbability_i)] + λ)
                                = (0.5 + 0.5) / [(0.5 x (1-0.5)) + (0.5 x (1-0.5)) + 0]
                                = 1  / 2 x (0.5 x 0.5)  = 2

            right top leaf output value with λ = 0:
               output value = (∑ Residuals_i)  / (∑ [PreviousProbability_i x (1 - PreviousProbability_i)] + λ)
                                = (-0.5) / [(0.5 x (1-0.5)) + 0]
                                = -0.5  / (0.5 x 0.5)  = -2

      step 5: Make predictions
          - make new prediction by starting with initial prediction, but we need to convert this probability
            to a log(odds) value

                odds = p / (1 - p)   where p is the probability

           initial output = log(odds) = log_e[ p / (1 - p)]
                          = log_e [ 0.5 / (1 - 0.5)] = log_e[1] = 0

        - just like Gradient Boost, XGBoost makes new predictions by starting with the initial Predictions
          and adding the output of the Tree scaled by the learning rate, eta (η) 
        - default learning rate η = 0.3 

        - left bottom leaf prediction (for observation with dosage = 2):
             log(odds for dosage =2) prediction = initial output + η x output value = 0 + 0.3 x -2 = -0.6 
           convert to probability:
             probability = e**log(odds) / [ 1 + e**log(odds)] =  e**-0.6 / [ 1 + e**-0.6] = 0.35 

             Note: the new residual has been reduceded for 0.5 to 0.35, which is .15 smaller

        - right bottom leaf prediction (for observation with dosage = 8):
             log(odds for dosage = 8) prediction = initial output + η x output value = 0 + 0.3 x 2 = 0.6 
           convert to probability:
             probability = e**log(odds) / [ 1 + e**log(odds)] =  e**0.6 / [ 1 + e**0.6] = 0.65 

             Note: the new residual has been reduced 0.5 to (1 - 0.65) = 0.35, which is .15 smaller


        -  If your initial prediction uses 0.5, the log_e(odds) will always equal '0'. However, you
           can change the initial prediction to any probability. if so,it will not be 0
           (default initial prediction: 0.5)


      step 6: build 2nd tree based on the new residuals

           - Note: calculating the Similarity Scores for the 2nd tree is a little more interesting because
             the previous Probabilities are no longer the same for all the observations

                                          Dosage < 5                           
                                         T/         \F
                                      -0.35       Dosage < 15                               
                                                /       \         
                                           0.35,0.35   -0.35                    

             root Similarity score = (∑ Residuals_i)**2  / (∑ [PreviousProbability_i x (1 - PreviousProbability_i)] + λ)
                  =  (0.35)**2 / [(0.35 x (1 - 0.35)) + (0.65 x (1 - 0.65)) + (0.65 x (1 - 0.65))  + (0.35 x (1 - 0.35)) + λ] 


      step 7: Make another tree based on residuals from previous tree - repeat steps 2 - 6
          - new predictions uses output values from both trees
            initial effectiveness + η x (output_tree1 + output_tree2 + ... + output_treeN)

          - repeat this step until residuals are super small or reach maximum number of trees


  In Summary, when building XGBoost Trees for Classification
    - calculated Similarity Scores for each leaf
        Similarity score = (∑ Residuals_i)**2  / (∑ [PreviousProbability_i x (1 - PreviousProbability_i)] + λ)
    - calculate gain score between child nodes and parent node to determine how to split the data
         Gain = child_Left_similarity + child_Right_similarity - parent_similarity
    - prune the tree by calculating the difference between Gain values and a user defined 'Tree complexity
      Parameter, γ (gamma)
          - if 'Gain - γ < 0',  remove branch, else do not remove branch 
          - if branch, is not remove, then the branch'es parent node will not be remove even if 'Gain - γ < 0' 
    - calculate the output values for the leaves
         output value = (∑ Residuals_i)  / (∑ [PreviousProbability_i x (1 - PreviousProbability_i)] + λ)

    - λ (lambda) is a regularization parameter and when λ > 0, it results in more pruning, by shrinking the
      Similarity Scores, and smaller Output Values for the leaves

    - the mininum number of Residuals in a leaf is related to a metric called 'Cover', which is the denominator
      of the Similarity Score minus λ (lambda)
                    cover = ∑ [PreviousProbability_i x (1 - PreviousProbability_i)]


--------------------------------------------------
XGBoost Part 3 (of 4): Mathematical Details
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=ZVFeW798-2I

  - For both Regression and Classificaiton, we built the trees using 'similarity Scores' and calculated
    the output values for the leaves
  - the only difference betweeen Regression and Classification is the 'Loss Function'


  - Gradiant Boost Regression Loss Function

      loss function = L(y_i, p_i) =  1/2(observed - predicted)**2 
               where: y_i: observed value for observation i
                      p_i: predicted value for observation i

         apply loss function to the initial predictions: 
             ∑ L(y_i, p_i) = ∑ L[(1/2)(y_i - p_i)**2]


  - Gradiant Boost Classification Loss Function (negative log-likelihood)

      loss function = L(y_i, p_i) =  -[(y_i)log(p_i) + (1 - y_i)log(1 - p_i)]
           note: p_i is replaced with p_i_0 (1st tree prediction)

  - XGBoost uses the Loss Functions to build trees by minimizing this equation
       ∑ L(y_i, p_i) + (1/2) λ O_value**2 
       - 1st part is the loss function
       - 2nd part is Regularization term - this term is similar to Ridge Regression term
       - the goal is to find an Output Value (O_value) for the leaf that minimizes the whole equation


   - XGBoost uses the 2nd Order Tayler Approximation to simplify the the math when solving for the
     optimal Output Value  for both Regression and Classification
       - out of scope for this statquest

         L(y,p_i + O_value) ~=  L(y,p_i) + [(d/d p_i) L(y,pi)] O_value = (1/2) [(d**2/d p_i**2) L(y,pi)] O_value**2
            - Loss function + 1st derviative of the Loss function + 2nd derivative of the Loss Function


--------------------------------------------------
XGBoost Made Easy | Extreme Gradient Boosting | AWS SageMaker
  https://www.youtube.com/watch?v=PxgVFp5a0E4

  XGBoost: Steps:
    - repeatedly builds new models and combine them into an ensemble model
    - initially build the first model and calculate the error for each observation in the dataset
    - Then you build a new model to predict those residuals (errors)
    - Then you add prediction from this model to the ensemble of models
    - XGBoost is superior compare to gradient boost algorithm since it offers a good balance between
      bias and variance (Gradient boosting only optimized for the variance so tend to overfit training
      data while XGBoost offers regulation terms that can improve model generalization)

                                                  
                           Calculate the errors   
      initial model   ---> based on the previous --->   Build a model to
      (starting point)     model (residuals)            to predict those errors --|
                                           ^                                      |
                                           |                                      |
                                           ------------  add last model to the <--|
                                                         ensemble


  Algorithm:
    - works by building a tree based on the error (residuals) from the previous tree
    - scales the trees and then adds the predictions from new tree to the prediction from the previous trees
    - Example adopted from Gradient Boost Part 1 (of 4): Regression Main Ideas by StatQuest with Josh Starmer
      https://www.youtube.com/watch?v=3CC4N4z3GJc&t=87s

          height   Favorite    Gender   Weight   
           (m)      color                (kg)    
        ---------------------------------------- 
           1.6       Blue        Male     88     
           1.6       Green      Female    76     
           1.5       Blue       Female    56     
           1.8       Red         Male     73     
           1.5       Green       Male     77     
           1.4       Blue       Female    57     
                                     ave:71.2


--------------------------------------------------
Deploying an XGBoost model with Sagemaker for regression then tuning the hyperparameters.
  Hands-on AI
  https://www.youtube.com/watch?v=AFeviL0Rhs0


--------------------------------------------------
XGBoost in Python from Start to Finish
  StatQuest with Josh Starmer
  https://www.youtube.com/watch?v=GrJP9FLV3FE


--------------------------------------------------
--------------------------------------------------
 Check out:
 Statquest: ROC and AUC, Clearly explained

--------------------------------------------------
Deploying an XGBoost model with Sagemaker for regression then tuning the hyperparameters.
  Data Science Solutions
  https://www.youtube.com/watch?v=AFeviL0Rhs0
--------------------------------------------------
Clustering with K-Means and Amazon SageMaker (ML Series)
  AWS User Group India
  Suman Debnath
  https://www.youtube.com/watch?v=at-cOSjXOQE
-------------------------------------------------
Machine Learning Python | Principal Component Analysis (PCA) in AWS Sagemaker [ Project1]
  SkillCurb
  https://www.youtube.com/watch?v=RjrfD1aVeXc

  Summary:
    - In this video we will teach you how to perform Principal Component Analysis in AWS Sagemake over US-Census 
      data in Python. Principal component analysis, or PCA, is a statistical procedure that allows you to summarize 
      the information content in large data tables by means of a smaller set of “summary indices” that can be more 
      easily visualized and analyzed.

    - The lab will cover the following steps

       1: Data Import
       2: Exploratory Data Analysis
       3: Data Modeling 
       4: Accessing PCA Model Attributes
       5: Model Deployment and Conclusion 

--------------------------------------------------
