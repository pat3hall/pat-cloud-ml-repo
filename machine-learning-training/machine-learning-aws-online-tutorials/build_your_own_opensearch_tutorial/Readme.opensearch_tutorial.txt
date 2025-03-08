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
     version: 2.3 [I used 2.5],
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

