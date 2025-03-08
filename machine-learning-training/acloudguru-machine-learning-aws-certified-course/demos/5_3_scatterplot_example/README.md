5.3 Seeing Your Data Relationships

  Resources:

    matplotlib Scatter plot
      https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

      matplotlib.pyplot.scatter (x, y, ...)
        - A scatter plot of y vs. x with varying marker size and/or color.

    matplotlib Line Charts
      https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

      matplotlib.pyplot.plot(*args, scalex=True, scaley=True, data=None, **kwargs)[source]
       - Plot y versus x as lines and/or markers.

    Kaggle Swedish Insurance dataset
      https://www.kaggle.com/datasets/redwankarimsony/auto-insurance-in-sweden?resource=download


  Relationships
    - visualizing relationships in your data can provide a good general overview, show distrbution,
      and correlation between attributes
    - visualizaint relationships can also help find outliers and extreme values

  Plot types:
    Scatter Plots (scatter charts)
      - graphs plot points along the x and you axis for two values
    Bubble Plots (bubble charts)
      - graphs plot points along the x and y axix for 3 values where bubbles size is the 3rd value

  Use Cases
    Scatter Plots (scatter charts)
      - Example: Is there any relationship between the size of a home and the price?
      - shows the relationships between 2 values
    Bubble Plots (bubble charts)
      - Example: Is there any relationship between the size of a home, the age of the home, and
      - shows the relationships between 3 values
        the price?

  Correlation:
    - use scatter plots to see if there is a:
       - Positive Correlation
         - increasing scatter line
       - Negative Correlation
         - decreasing scatter line
       - No Correlation
         - random values with no visual correlation

   Code: swedish-auto-scatter plot example (demos/5_3_scatterplot_example/scatterplot_example.ipynb)

         >>> # set dir
         >>> cd "pat-cloud-ml-repo/machine-learning-training/acloudguru-machine-learning-aws-certified-course/demo/5_3_scatterplot_example"

         >>> # import python libraries
         >>> # %matplotlib inline
         >>> import sys
         >>> import numpy as np
         >>> import pandas as pd
         >>> import matplotlib.pyplot as plt

         >>> # read in data
         >>> df = pd.read_csv ("swedish_insurance.csv")
         >>> df.head

         >>> # plot graph
         >>> plt.scatter(x=df['X'], y=df['Y'])
         >>> plt.title('Scatter Plot Example')
         >>> plt.xlabel('Number of Claims')
         >>> plt.ylabel('Total Payed Out (in thousands)')
         >>> plt.savefig('Scatter_Plot_Example.png', format='png', dpi=1200)
         >>> plt.show()


Summary:

  scatter plots (charts) vs bubble plots (charts):
    - scatter plots show the relationship between two attributes,
    - bubble plots show the relationship between three attributes.


