-----------------------------------------
  AWS Kendra
-----------------------------------------


Using Amazon Kendra to Build your Enterprise Product Search
  ML Workbench
  https://www.youtube.com/watch?v=YuF5daC7TRQ

  Summary:
    In this video, we'll use Amazon Kendra build a product search capability.  This includes:
    1. Downloading and processing an e-commerce dataset
    2. Creating a Kendra index
    3. Connecting a data source
    4. Tuning our search results
    5. and finally, using the python SDK to programmatically query our index.

   Links:
      The notebooks I used in this tutorial is available here:
      https://github.com/chaeAclark/literate-eureka/tree/main/aiservices/kendra

   The dataset used for this tutorial is available here:
      https://www.kaggle.com/datasets/mewbius/ecommerce-products
      eCommerce data scraped from the L.L. Bean website. 
         productsclassified.csv 
           - a small sample of the data with an extra column with the classification (an assigned value 
             that represents one of lowest nodes of a branch of a product taxonomy).


   Dataset
     - kaggle LL Bean ecommerce data (productsclassified.csv)
     - download from Kaggle, upload to S3 bucket 
     - to process (convert to JSON), will use Jupyter Notebook in SageMaker

1. Convert CSV Data to JSON data (1 row per json file) and JSON Metadata and upload to S3 Bucket 

     AWS -> SageMaker -> Notebook -> start -> Jupyter 
        -> Upload to Jupyter Notebook: 
                  productsclassified.csv
                  ProcessProductData.ipynb
                  create folder named "jsons"
       -> Open ProcessProductData.ipynb
          - 1 convert CSV to DataFrame only keeping columns: 'name','description','itemid','colorname','Classification'
          - remove quotes and brackets from itemid field, and place '0' in empty 'itemid' fields
          - convert colorname from a list - remove quotes and brackets
          - convert description from a list - remove quotes and brackets
          - rename columns from:to  : 'name':'product','description':'description','itemid':'id',
                                      'colorname':'colors','Classification':'category'
          - add new column called "user_profile" and randomly add values of reseller', 'consumer', or 'distributer'
          - save Dataframe as JSON doc and metadata json doc locally under jsons, and then upload to S3 bucket
            - each row save to a separate json file contaiing just id, product, & description fields, save to jsons/doc_<rowid>_prod_<id>.json
            - each row save to a separate metadata json file containing document id (json file name),  and Attributes where
              attributes include _category (category), colors, and user_profile, save to jsons/doc_<rowid>_prod_<id>.metadata.json 
            - upload metadata files to S3, placed in 'metadata' subfolder

       Note: Kendra Metadata 
            - provide means to filter and sort your json data based on the attributes in the metadata (e.g. by color, user_profile, etc.)

      JSON data file, 'doc_0_prod_297896.json', for row 0:
       {
           "id": "297896",
           "product": "Men's Vasque Talus Trek Waterproof Hiking Boots",
           "description": "These waterproof hiking boots for men are rugged enough for peak performance yet light and quick enough to keep feet from feeling weighed down."
       }

      JSON metadata file, 'doc_0_prod_297896.metadata.json', for row 0:
       {
           "DocumentId": "doc_0_prod_297896.json",
           "Attributes": {
               "_category": "Boots",
               "colors": [
                   "Slate Brown/Chili Pepper"
               ],
               "user_profile": "consumer"
           },
           "Title": "Men's Vasque Talus Trek Waterproof Hiking Boots"
       }


2. Create Kendra Index


  AWS -> Kendra -> Create Index ->
    Index Name: product-index, 
    IAM : Create a new role, Role Name: AmazonKendra-us-east-1-kendraproductindex
    -> Next ->
    Edition: Developer edition
    -> Next -> Review -> Create
    # takes ~30 sec 


Facet 
  - Information about a document attribute or field. You can use document attributes as facets.
  - An array of document attributes that are nested facets within a facet.
  - For example, the document attribute or facet "Department" includes the values "HR", "Engineering", and 
    "Accounting". You can display these values in the search results so that documents can be searched by 
    department.


3. Add facet definitions 
   - enabled metadata to be to search, sort, and as display fields

  AWS -> Kendra -> Indexes -> product-index -> Facet Definition -> 
     Note: _category is a default facet index field (type: string, Useage Type: Sortable & Searchable)
     -> Add field -> Field Name: colors      , Data type: String List, Usage Types: Facetable & Searchable & Displayable 
         -> add
     -> Add field -> Field Name: user_profile, Data type: String     , Usage Types: Facetable & Searchable & Displayable 
         -> add

4. Add Custom Datasource


  Kendra Data Source:
    - A data source contains a connector that establishes a connection between the documents stored in your 
      repository and your Amazon Kendra index. 
    - The data source uses the connector to watch your document repository and determine which documents 
      need to be indexed, re-indexed, or deleted.
    - Use Kendra's connectors for popular sources like file systems, web sites, Box, DropBox, Salesforce, 
      SharePoint, relational databases, and S3, and more

      Note: Kendra data source includes Option for sample S3 Data source (ocvers kendra, EC2, S3, and Lambda)

  Kendra Data source Sync mode
    - Choose how you want to update your index when your data source content changes.
    Full sync
      - Sync and index all contents in all entities, regardless of the previous sync status.
    New, modified, or deleted content sync
      - Only sync new, modified, or deleted content.

  Kendra Data Source Sync run schedule
    - Tell Amazon Kendra how often it should sync this data source. 
    - You can check the health of your sync jobs in the data source details page once the data source is 
      created.
    Frequency
       Run on Demand, hourly, daily, weekly, monthly, or custom


  AWS -> Kendra -> Indexes -> product-index -> Add Data Source ->  
     Available Data sources -> Amazon S3 connector -> Add connector ->
       Data source name: pat-demo-bkt -> Next ->
       IAM : Create a new role, Role Name: AmazonKendra-productindexdatasource
       -> Next -> 
        Enter the data source location:  s3://pat-demo-bkt
        Metadata files folder location: metadata/
        max file size <default> 50 MB
        Sync Mode: Full sync
        Sync Schedule: Run on demand
        -> Next -> set Field Mappings - optional 
        -> Next -> Review -> Add Data source
        # crawl on your S3 data source
        -> sync Now 
        # for 1 doc, ~5 min
        -> examine sync history

5. Kendra Test Search

  AWS -> Kendra -> Indexes -> product-index -> Search Index content <left tab>

      search: I would like hiking boots
        -> shows content from index docs

        shows 43 results and first result is:

           Men's Vasque Talus Trek Waterproof Hiking Boots
           ...id": "297896", "product": "Men's Vasque Talus Trek Waterproof Hiking Boots", "description": 
            "These waterproof hiking boots for men are rugged enough for peak performance yet light and quick 
            enough to keep feet from feeling weighed down...
           
           https://pat-demo-bkt.s3.amazonaws.com/doc_0_prod_297896.json
           
           Document fields
             _source_uri
             {"StringValue":"https://pat-demo-bkt.s3.amazonaws.com/doc_0_prod_297896.json"}
             
             user_profile
             {"StringValue":"consumer"}
             
             s3_document_id
             {"StringValue":"doc_0_prod_297896.json"}
             
             colors
             {"StringListValue":["Slate Brown/Chili Pepper"]}
             

        Notes: 
            - document fields are based on the facet indexes
            - "filter Search result" icon include facet searchable fields (colors & user_profile)
            - if index contect was a web site, it would show website content
            - if json data includes a URL, it would display URL

  Update Facet definition to add "facetable" to field name: "_category"
     -> save

      search: I would like boots
        -> shows content from index docs
            - "filter Search result" icon now also include facet fields (colors, user_profile, & _category)

            Relevance Tuning
              click on slider icon <right side> 
  Update Facet definition to add "Searchable" to field name: "_category"
     -> save
         Relevance Tuning
           -> now includes category field


5. Kendra API Query from Jupyter Notebook

   To SageMaker Notebook Instance Role, add: AmazonKendraReadOnlyAccess
     AWS -> SageMaker -> Notebook -> start -> open Jupyter 
        -> Upload to Jupyter Notebook: 
                  KendraAPICalls.ipynb
       -> Open KendraAPICalls.ipynb
          -> Under "Query Parameters", modify "index_id = "...." to use 'product-index' index ID value


    Code: Kendra API Code (from KendraAPICalls.ipynb)

        import boto3

        # #### Methods

        def display_results(response:dict, user_profile:str=None) -> None:
            list_1 = []
            list_2 = []
            for i,item in enumerate(response['ResultItems']):
                title = item['DocumentTitle']['Text']
                score = item['ScoreAttributes']['ScoreConfidence']
                for attr in item['DocumentAttributes']:
                    if (attr['Key'] == 'user_profile'):
                        if  attr['Value']['StringValue'] == user_profile:
                            list_1.append(f'{i}. [{score}] [{attr["Value"]["StringValue"]}] {title}')
                        else:
                            list_2.append(f'{i}. [{score}] [{attr["Value"]["StringValue"]}] {title}')
                        break
                    else:
                        continue
            results = list_1 + list_2
            _ = [print(item) for item in results]


        # ## Query Parameters
        kendra = boto3.client("kendra")

        index_id = "c6b5e9c5-9d95-4385-bfa9-e1d2412fab16"
        query = "boots please"
        user_profile = "consumer"


        # ## Query Index
        response = kendra.query(
            QueryText = query,
            IndexId = index_id
        )


        # #### Response Example
        response.keys()

        Out[7]: dict_keys(['QueryId', 'ResultItems', 'FacetResults', 'TotalNumberOfResults', 'ResponseMetadata'])


        response['TotalNumberOfResults']
        Out[8]: 20

        response['ResultItems'][0]

        Out[9]:
        {'Id': '0ab8786e-ac9f-441c-89d2-468be5a05b83-7ac44a66-606d-477b-a7f4-efe7f86f037f',
         'Type': 'DOCUMENT',
         'Format': 'TEXT',
         'AdditionalAttributes': [],
         'DocumentId': 's3://pat-demo-bkt/doc_137_prod_175051.json',
         'DocumentTitle': {'Text': 'Men\'s Bean Boots by L.L.Bean®, 6"',
          'Highlights': [{'BeginOffset': 11,
            'EndOffset': 16,
            'TopAnswer': False,
            'Type': 'STANDARD'}]},
         'DocumentExcerpt': {'Text': '...id": "175051",\n    "product": "Men\'s Bean Boots by L.L.Bean\\u00ae, 6\\"",\n    "description": "Warm...
         ....
         ....
         ....
          }
         }

        # ### Results by Relevance
        display_results(response)

        Out[10]: 
        0. [HIGH] [distributer] Men's Bean Boots by L.L.Bean®, 6"
        1. [HIGH] [distributer] Kids' Northwoods Boots
        2. [HIGH] [consumer] Men's Vasque Talus Trek Waterproof Hiking Boots
        3. [HIGH] [reseller] Kids' Bogs® Classic High Handles Boots
        4. [HIGH] [consumer] Kids' Bogs Boots, Classic Camo
        5. [HIGH] [distributer] Men's Keen Targhee Waterproof Hiking Boots, Insulated
        6. [MEDIUM] [consumer] Men's L.L. Bean Boots, 10" Shearling-Lined
        7. [MEDIUM] [distributer] Keen Elsa WP Boots
        8. [MEDIUM] [consumer] Ultralight Waterproof Pac Boots, Tall
        9. [MEDIUM] [reseller] Men's Wicked Good Moc Boots II

        # ### Results by User Profile
        display_results(response, user_profile)

        Out[11]:
        2. [HIGH] [consumer] Men's Vasque Talus Trek Waterproof Hiking Boots
        4. [HIGH] [consumer] Kids' Bogs Boots, Classic Camo
        6. [MEDIUM] [consumer] Men's L.L. Bean Boots, 10" Shearling-Lined
        8. [MEDIUM] [consumer] Ultralight Waterproof Pac Boots, Tall
        0. [HIGH] [distributer] Men's Bean Boots by L.L.Bean®, 6"
        1. [HIGH] [distributer] Kids' Northwoods Boots
        3. [HIGH] [reseller] Kids' Bogs® Classic High Handles Boots
        5. [HIGH] [distributer] Men's Keen Targhee Waterproof Hiking Boots, Insulated
        7. [MEDIUM] [distributer] Keen Elsa WP Boots
        9. [MEDIUM] [reseller] Men's Wicked Good Moc Boots II



  Clean up
      AwS -> Kendra -> indexes -> product-index -> delete

