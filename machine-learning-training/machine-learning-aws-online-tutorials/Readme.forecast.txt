------------------------------------------
 Amazon Forecast
------------------------------------------
NOTE:
Amazon Forecast is no longer available to new customers. Existing customers of Amazon Forecast can 
   continue to use the service as normal.Learn more. 
------------------------------------------

Amazon Forecast Tutorial | How to use Amazon Forecast Service | AWS Forecast Demo using Retail Data
  https://www.youtube.com/watch?v=LHn4-l2xWaQ

  Retail forecast Dataset:
    https://archive.ics.uci.edu/dataset/352/online+retail

  Forecast Use Cases:
    Weather
      - Weather forecasting
    Banking
      - loan / Credit card account open forecasting
    housing
    - housing prices forecasting, etc.
    workforce planning
      - forecast workforce staffing at 15 min increments to optimize for high and low periods
    Retail and inventory planning
      - reduce waste, increase inventory turns, and improve in-stock availability by forecasting
        product demand at specific probability levels
    Travel demand forecasting
      - forecast foot traffic, visitor counts, and channel demand to more efficiently manage
        operating costs


  Forecast - how it works

                                                                             |--> Visualize forecast
     Historical Data               Upload data                               |
     - Sales, web traffic,      -------| to S3                               |
       inventory, cashflow, etc        |          Amazon       Customized    |
                                       |--------> Forecast     Forecasting --|--> export to CSV
                                       |                       Model         |
                                       |                                     |    Retreive Using
     Related Data               -------|                                     |--> Forecast API
     - Holidays, product 
       descriptions, 
       promotions, etc
 
 
  Amazon Forecast supports the following dataset domains:
    RETAIL Domain 
      – For retail demand forecasting
    INVENTORY_PLANNING Domain 
      – For supply chain and inventory planning
    EC2 CAPACITY Domain 
      – For forecasting Amazon Elastic Compute Cloud (Amazon EC2) capacity
    WORK_FORCE Domain 
      – For work force planning
    WEB_TRAFFIC Domain 
      – For estimating future web traffic
    METRICS Domain 
      – For forecasting metrics, such as revenue and cash flow
    CUSTOM Domain 
      – For all other types of time-series forecasting


  Forecast Implementation
    Dataset Groups
      - import data, metadata or related data from S3
    Create Predictor
    Create Forecast - Query Forecast
    Create Forecast Export (CSV)


    Dataset & Use Case
      Dataset
        - https://archive.ics.uci.edu/dataset/352/online+retail
        - this dataset contains all transactions between 01/12/2010 and 0/12/2011 for a UK based
          retailer, which sell unique all-occasions gifts
       Usecase:
         - predict demand for different items sold by the stores for next 60 days


  step 1: Create Dataset Group [import dataset]

   AWS -> Forecast -> Dataset group -> Create dataset group ->
      Dataset Group Name: retail_dsg, Forecast domain: retail -> Next
       Frequency of your data: 1 min, 
       Data Schema: Schema Builder: 
         # required attributes from your chosen domain [retail]

            Attribute Name: item_id,   Attribute Type: string
            Attribute Name: timestamp, Attribute Type: timestamp
            Attribute Name: demand,    Attribute Type: float

            # modified Xlsx columns to:
              StockCode -> item_id, Quantity -> demand
            # modified xlsx column and reformated 
              InvoiceDate -> timestamp: reformatted to: yy-mm-dd hh:mm:ss
            # removed columns:
              CustomerID, Country, Description
            # exported as CSV

            Add attribute:
            Attribute Name: unitprice,    Attribute Type: string  # by default, new attributes are strings

            Dataset import name: retail_ds_import
            select time zone: do not use time zone
           
            # import file types CSV or Parquet
            Import file type: CSV 

            S3 location: <bucket>
            IAM Role: Create IAM Role
            -> Start

            # creates dataset # ~30 min
            # options to import [upload] item metadata data and 
            #  Related time series data [e.g. promotions]

  step 2: Create Predictor [train model]

            # after data is imported, 
              Train a predicter -> Start
                 Predictor Name: retail_predictor
                 Forecast frequency: 1 day
                 # how far in the future to forecast [e.g. 60 days]:
                 forecast horizon : 60 

             # Forecast quantiles: [up to 5 - default 3: 0.10, 0.50, 0.90
             #   quantiles: chance your actual value will be below the predicted value 

             -> Create
               # may take 1-1/2 to 2 hours to train the model

  step 3: Create Forecast - Query Forecast

       AWS -> Forecast -> Dataset Group -> retail_dsg -> Create forecast -> 

         forecast name; retail_forecast, predictor: <predictor>, forecast quantiles: 0.1, 0.5. 0.9
         Items for generating forecast: All items
         -> Start
         # takes ~ 1 hour



       AWS -> Forecast -> Dataset Group -> retail_dsg -> Query Forecast -> 
         # select predictor:
         Forecast type: retail_forecast
         # select dates
         Start date: 2011/12/09   end date: 2012/02/07
         item_id: 71053
         -> get forecast
            -> shows graph for each quantile

  step 4: Create Forecast Export (CSV)

       AWS -> Forecast -> Dataset Group -> retail_dsg -> Forecast -> select "retail_forecast" -> 
          Create forecast export -> 
             Export Name: retail_forecast_export, IAM Role: <previously created role> ,
             # format options are CSV and Parquet
             Format: CSV , bucket: <bucketName>
             -> start
             # could take 1/2 hour

