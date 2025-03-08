------------------------------------------------------
3.9 Performing Feature Engineering Using Amazon SageMaker Lab


Perform Feature Engineering Using Amazon SageMaker

About this lab

Imagine you are the data engineer, and you have been assigned the task of preprocessing the data and getting it
ready for the machine learning engineers to create a highly predictable model. Your data contains both text and
numerical data. The numerical data is of different ranges, and some text features require proper ordering.

In this hands-on lab, you will learn how to encode, scale, and bin the data using scikit-learn.
  - Learning objectives
  - Launch SageMaker Notebook
  - Load Libraries and Prepare the Data
  - Apply Encoding Techniques
  - Apply Scaling Techniques
  - Apply Binning Techniqu


    ------------------------------------------------------

Solution
Launch SageMaker Notebook

    To avoid issues with the lab, open a new Incognito or Private browser window to log in. This ensures that your personal account credentials, which may be active in your main window, are not used for the lab.
    Log in to the AWS Management Console using the credentials provided on the lab instructions page. Make sure you're using the us-east-1 region. If you are prompted to select a kernel, please choose conda_tensorflow2_p310.
    In the search bar, navigate to Amazon SageMaker.
    Under the Notebooks section in the left menu, click Notebook Instances.
    Confirm the notebook is marked as InService. If so, click the Open Jupyter link under Actions.
    Click on the Feature_Engineering.ipnyb file.

Load Libraries and Prepare the Data

    Click the first cell that imports the required Python libraries to highlight it, and use the Run button at the top to execute the code.

    An asterisk inside the square braces, (In [*]), indicates the code is running. You will see a number, (In[1]), once the execution is complete. This first cell uses the Pandas library and reads the raw data from Employee_encoding.csv.

    The second cell requires you to write the code to read the input file and load the dataframe. Paste the following Python code and click Run.

    employee_df = pd.read_csv('Employee_encoding.csv')
    employee_df.head()

Apply Encoding Techniques

    In the next cell under 3.1) Ordinal ENcoding, you will initialize the OrdinalEncoder and perform the fit operation. Highlight this cell and click Run.

    The next cell uses the categories_ attribute to display the encoder's sequence. Highlight this cell and click Run.

    In the third cell, insert and Run the following Python code to assign the transformed values to a new feature named encoded_title.

    employee_df['encoded_title'] = ordinal_encoder.transform(employee_df['title'].values.reshape(-1,1))
    employee_df.head()

    In the first cell, under 3.2) One-hot Encoding, the code initializes the OneHotEncoder. Highlight this cell and Run it.

    In the next cell, use fit_transform to perform fit and transform in a single function call. Update the cell with the following Python code and Run it.

    transform = gender_encoder.fit_transform(employee_df['gender'].values.reshape(-1,1))

    In the third cell, use the todense function to address the sparse nature of the data and join the output with the parent dataframe. Highlight this cell and Run it.

    In the first cell, under 3.3) Label Encoding, paste and Run the following code to initialize LabelEncoder.

    department_encoder = LabelEncoder()

    The next cell applies fit and transform on the department feature and assigns the output to encoded_department. Highlight this cell and click Run.

Apply Scaling Techniques

    In the first cell under 4) Scaling Techniques, the code scales the salary feature using MinMaxScaler. Highlight this cell and click Run.

    In the next cell, paste and Run the below code to invoke the describe function on the salary_minmax_scaled feature and ensure the value ranges between 0 and 1.

    employee_df[['salary_minmax_scaled']].describe()

Apply Binning Techniques

    In the first cell under 5) Binning Techniques, paste and Run the following Python code to initialize KBinDiscretizer with ten bins.

    kbins = KBinsDiscretizer(n_bins=10, strategy='quantile', encode='ordinal')

    Highlight the next cell and Run it to invoke fit_transform on the Kbins discretizer.

    Highlight and Run the code in the last cell to visualize the new age_bin feature using Matplotlib's histogram function.


    ------------------------------------------------------


