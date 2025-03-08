###################################################
Demo: Zero to Hero with Amazon OpenSearch Service | Amazon Web Services
https://www.youtube.com/watch?v=wh2tn5BTBpg

  Prashant Agrawal
  Summary:
    - Discover how you can get started with Amazon OpenSearch Service--from the AWS Service Console
      all the way through Dashboards--in ~10 minutes.

  OpenSearch Service
    - fully managed service that makes it easy to deploy and scale open search cluster in AWS


  Opensearch use cases:

    Log Analytics
      - locate, diagnose, and remediate issues with your infrastructure and AWS wervices
      - improve your product's latency and stability
    Search
      - find the right product, service, document, or answer quickly across semi-structured and unstructured
        data and different facets and attributes

  How Opensearch works
    1. Send data as JSON or REST APIs
      data sources:
        - Server, application, network, AWS, and other logs
        - Application data
    2. Data is indexed [by Opensearch] - all fields searchable including nested JSON
       - everything is indexed, and everything is searchable
    3. REST APIs, for fielded matching, boolena expressions, sorting and analysis
      - user business user can log in to opensearch dashboard, anr run interactive visulization and transport

  Opensearch demo

     AWS -> Opensearch -> Domains <left tab> -> create Domain
        Custom Endpoint
          - domain has an auto-generated endpoint, but you can add a custom endpoint using AWS Certificate
            manager

        Deployment type:
          Production
            - workloads spanning multiple AZ and dedicated master
            - choice between 3 AZ and 2 AZ
          Deployment and testing
            - choice between 3 AZ, 2 AZ, and 1 AZ

       Data Nodes:
         - select compute Instance family (OR1, General Purpose, Compute Optimized, Memory Optimized,...
         - select instance type from familly
         - select number of nodesl
         - select [EBS] Storage Type and size

        ...

    -> Create

    # Domain created:
       select [domain] -> click on "OpenSearch Dashboard URL -> opens dashboard GUI
        -> log in
        -> Dev tools <Left tab>
        -> enter scripts to index your data
           PUT /my-movie-index
           {
             "settings": {
               "number_of_shards": 1,
               "number_of_replicas": 2,
             }
           }
        -> after data is index, you can enter search queries
           # find how many docs are in "my-movie-index"
           POST my-movie-index/_search
           # find movie titles with "iron" in "my-movie-index" index
           GET my-movie-index/_search
           {
             "query": {
               "term": {,
               "title": "iron",
             }
           }

        # back to dashboard
          -> create log traffic dashboards

###################################################
Elasticsearch vs. OpenSearch: 6 Key Differences and How to Choose
https://coralogix.com/guides/elasticsearch/elasticsearch-vs-opensearch-key-differences/

What Is AWS OpenSearch?
  - OpenSearch is an open source project created by AWS in 2021, as a fork of Elasticsearch 7.10.2.
  - This means it has the same basic functionalities as Elasticsearch, but since then the project has diverged
    from Elasticsearch in several ways.
  - Amazon provides a fully managed search and analytics service called AWS OpenSearch.
    It includes OpenSearch and OpenSearch Dashboards (a community-driven, open-source data visualization and
    user interface suite derived from Kibana 7.10).
  - Users pay only for the resources they run on AWS, with no additional charge for the search and
    visualization software.
###################################################

Build Your Own Search Using Amazon OpenSearch Service

Johnny Chivers
https://www.youtube.com/watch?v=SIl5PM4m2KM

  From Summary:
    opensearch
    - OpenSearch is a distributed, community-driven, Apache 2.0-licensed, 100% open-source search and analytics
      suite used for a broad set of use cases like real-time application monitoring, log analytics, and
      website search.
    - OpenSearch provides a highly scalable system for providing fast access and response to large volumes of
      data with an integrated visualization tool, OpenSearch Dashboards, that makes it easy for users to explore
      their data.
    - OpenSearch is powered by the Apache Lucene search library, and it supports a number of search and analytics
      capabilities such as k-nearest neighbors (KNN) search, SQL, Anomaly Detection, Machine Learning Commons,
      Trace Analytics, full-text search, and more. In this Video I take you through the steps to set up your
      first domain.

  OpenSearch
    - a full analytics and search engine managed service by AWS
    - AWS manages provisioning the nodes. ensuring they are healthy, installing the opensearch software,
      and managing the underlying infastructure
    OpenSearch features
      - attach 3 terabits ??? of memory.
      - UI where search can be handle via a GUI
    opensearch configuration
      - main node has data nodes attached
      - upload info/docs to be search to create indexes
      - search the indexes via API or uses UI

   master pw: <yourMasterPW>

   AWs -> Opensearch -> Create Domain ->
     Name: opensearch-tutorial, Deployment: Dev/Test, AZ: 3-AZ, [Standby],
     instance type: t3.small.search, # keeps in Free tier
     version: 2.3 [I used 2.15],
     # use 'public' and 'master user' to keep it simple for tutorial
     Network: Public, Master User: Create Master User, Master Username: tutorial-user, PW: <yourMasterPW>,
     Domain Access Policy: ONly use fine-grained access control
     -> Create
       -> takes 10 to 15 min

  Upload a signle document using the api

    curl -XPUT -u 'tutorial-user:os@Tut4me' 'https://search-opensearch-tutorial-rrnqunh2y5emx6x4b5capcrrxe.us-east-1.es.amazonaws.com/movies/_doc/1' -d '{"director": "Burton, Tim", "genre": ["Comedy","Sci-Fi"], "year": 1996, "actor": ["Jack Nicholson","Pierce Brosnan","Sarah Jessica Parker"], "title": "Mars Attacks!"}' -H 'Content-Type: application/json'


    curl -XPUT -u 'master-user:master-user-password' 'domain-endpoint/movies/_doc/1' -d '{"director": "Burton, Tim", "genre": ["Comedy","Sci-Fi"], "year": 1996, "actor": ["Jack Nicholson","Pierce Brosnan","Sarah Jessica Parker"], "title": "Mars Attacks!"}' -H 'Content-Type: application/json'


    Upload multiple documents create a json file and upload - upload "bulk_movies.json' to _blk

      curl -XPOST -u 'tutorial-user:os@Tut4me' 'https://search-opensearch-tutorial-rrnqunh2y5emx6x4b5capcrrxe.us-east-1.es.amazonaws.com/_bulk' --data-binary @bulk_movies.json -H 'Content-Type: application/json'

      curl -XPOST -u 'master-user:master-user-password' 'domain-endpoint/_bulk' --data-binary @bulk_movies.json -H 'Content-Type: application/json'

  Search for Document using API - search for movie with "mars" in its doc, return in 'pretty' format

    curl -XGET -u 'tutorial-user:os@Tut4me' 'https://search-opensearch-tutorial-rrnqunh2y5emx6x4b5capcrrxe.us-east-1.es.amazonaws.com/movies/_search?q=mars&pretty=true'

    curl -XGET -u 'master-user:master-user-password' 'domain-endpoint/movies/_search?q=mars&pretty=true'

    returned:
      {
        "took" : 943,
        "timed_out" : false,
        "_shards" : {
          "total" : 5,
          "successful" : 5,
          "skipped" : 0,
          "failed" : 0
        },
        "hits" : {
          "total" : {
            "value" : 1,
            "relation" : "eq"
          },
          "max_score" : 0.2876821,
          "hits" : [
            {
              "_index" : "movies",
              "_id" : "1",
              "_score" : 0.2876821,
              "_source" : {
                "director" : "Burton, Tim",
                "genre" : [
                  "Comedy",
                  "Sci-Fi"
                ],
                "year" : 1996,
                "actor" : [
                  "Jack Nicholson",
                  "Pierce Brosnan",
                  "Sarah Jessica Parker"
                ],
                "title" : "Mars Attacks!"
              }
            }
          ]
        }
      }


  Search for document via UI Search Via UI

      domain-endpoint/_dashboards/

      -> click on "OpenSearch Dashboards URL
        -> login "tutorial-user" ...
        -> Explore on my own
        -> tenant: Global
        -> <hamburger> <top-left> -> Discover
          -> Create Index pattern -> Index pattern name: movies* -> Next Step -> Create index pattern
          # shows field names that can be search on
        -> <hamburger> <top-left> -> Discover
          # shows index of movies info that was uploaded
          -> search by enter key words in top search bar,e.g. Marshals
          -> seach for movie with "lee" and "jones":  lee and jones


   # clean up
     AWS -> Domains -> opensearch-tutorial -> Delete


###################################################
Setting Up a Amazon Opensearch (ElasticSearch) Cluster with Free Tier
  https://www.youtube.com/watch?v=BNOYTbRbaFQ

  opensearch Query DSL (domain-specific language) documentation
    https://opensearch.org/docs/latest/query-dsl/

  summary:
   In this video, I walk you through the process of setting up your first OpenSearch cluster and viewing sample
   data in the Kibana dashboard. Everything done in this video is using the AWS Free Tier.


  AWS -> Opensearch -> Get Started
       Options:
          Managed clusters: Deploy a secured OpenSearch cluster in minutes.
          Serverless :      Start working with your data with a simple, scalable collection.
          Ingestion:        Create an ingestion Serverless pipeline to filter, enrich, transform, and
                            aggregate data and deliver to an OpenSearch Service domain.
                            - setup a pipeline of data from cloudtrail or s3, etc

     Managed Cluster -> Create Domain ->

     Name: os-domain, Domain Creation Methdo: Standard Create, Deployment: Dev/Test,
     Domain without Standby, AZ: 1-AZ,
     instance type: t3.small.search, number of nodes: 1, EBS storage size per node: 10 # keeps in Free tier
     version: 2.7 [I used 2.15],
     # dedicated Master nodes: use for traffic routing, replication, maintenance task - offloads task from data nodes
     #  data nodes: host your data and serving real traffic
     #   -> dedicated Master nodes -> NOT enabled
     #   if enabled, select instance type and number of nodes (e.g. 3 or 5)
     # use 'public' and 'master user' to keep it simple for tutorial

     # custom endpoints: if you have domain that you setup as part of AWS,  you can specify custom hostname
     #     and AWS certificate
     # else: AWS will assign a random domain for you to use

     # Network: VPC -> allows you to keep your opensearch instances private (not exposed to public internet)
     #   normally in a VPC in a private subnet without inbound internet access with access from an EC2
     #   instance hosting traffic in a public subnet

     Network: Public,
     Enable fine Grain control, Master User: Create Master User, Master Username: admin, PW: <yourMasterPW>,
     Domain Access Policy: ONly use fine-grained access control

     # Amazon Cognito autentication: Enable to use Amazon Cognito authentication for OpenSearch Dashboards/Kibana.
     # NOT enabled

     # Access policy: Access policies control whether a request is accepted or rejected when it reaches the
     #    Amazon OpenSearch Service domain. If you specify an account, user, or role in this policy, you must
     #    sign your requests.
     Domain Access Policy: Only use fine-grained access control
     -> Create
       -> takes ~15 min
      -> to interactive programmatic uses Domain endpoint

      -> to UI, use "OpenSearch Dashboards URL"
      -> click on "OpenSearch Dashboards URL"
        -> login "tutorial-user" ...
        -> Add data # to add sample data
        -> tenant: Global -> Confirm
        # sample data options: eCommerce orders, Flight Data, Web Logs
        -> Sample Flight Data -> Add -> View Data
           -> gives a default dashboard for the sample data
           -> to create your own dashboard -> <hamburger> <top-left> -> Dashboard
               -> dashboard widgets created with a drag and drop interface

        # when you interactive with opensearch, you are using DSL (domain specific language)

       -> Total Flights [widget] -> 3 dots <top right corner> -> inspect -> View Data -> Requests ->
          Statistics <left tab>: shows request statitics on the
          request <center tab>: shows DSL request json set to opensearch
          response <left tab>: shows DSL response json to the request

        # each widget is are independent queries pulling data, and then visualizing the data

     -> <hamburger> <top-left> -> discover ->
        # show raw data, and allows you to search it
        # essentially default kibana view

     # interactive with DSL
     -> <hamburger> <top-left> -> Overview  -> Interactie with Opensearch API:
       -> in left pane, paste search DSL request (copied from Flight count widget) then click run button <upper left>

        GET _search
        {
          "aggs": {},
          "size": 0,
          "stored_fields": [
            "*"
          ],
          "script_fields": {
            "hour_of_day": {
              "script": {
                "source": "doc['timestamp'].value.hourOfDay",
                "lang": "painless"
              }
            }
          },
          "docvalue_fields": [
            {
              "field": "timestamp",
              "format": "date_time"
            }
          ],
          "_source": {
            "excludes": []
          },
          "query": {
            "bool": {
              "must": [],
              "filter": [
                {
                  "match_all": {}
                },
                {
                  "range": {
                    "timestamp": {
                      "gte": "2024-09-08T14:56:56.470Z",
                      "lte": "2024-09-20T19:00:00.000Z",
                      "format": "strict_date_optional_time"
                    }
                  }
                }
              ],
              "should": [],
              "must_not": []
            }
          }
        }

        -> right pane displays the response
        {
          "took": 204,
          "timed_out": false,
          "_shards": {
            "total": 25,
            "successful": 25,
            "skipped": 0,
            "failed": 0
          },
          "hits": {
            "total": {
              "value": 3759,
              "relation": "eq"
            },
            "max_score": null,
            "hits": []
          }
        }


   # clean up
     AWS -> Domains -> os-domain -> Delete
       -> takes ~15 min

###################################################
