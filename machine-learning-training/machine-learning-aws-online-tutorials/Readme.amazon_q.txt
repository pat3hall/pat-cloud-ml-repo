-------------------------------------------
  Amazon Q Developer & Amazon Q Business
-------------------------------------------
  Amazon Q Developer
    - Amazon Q Developer assists developers and IT professionals with all their tasks—from coding,
      testing, and upgrading applications, to diagnosing errors, performing security scanning and fixes,
      and optimizing AWS resources.
    - Amazon Q has advanced, multistep planning and reasoning capabilities that can transform (for example,
      perform Java version upgrades) and implement new features generated from developer requests.

    Q developer Free Tier:
      - Code faster with code suggestions in the IDE and CLI
      - Free for public CLI completions
      - Review code licenses with reference tracking
      - Use where you work: your IDE, CLI, the AWS Management Console, Slack, and more
      - Limited monthly access of advanced features:

        - Chat, debug code, add tests, and more in your integrated developer environment (IDE)
         (limit 50 interactions per month)
        - Accelerate tasks with the Amazon Q Developer Agent for software development (limit 5 per month)
        - Upgrade apps in a fraction of the time with the Amazon Q Developer Agent for code transformation
          (limit 1,000 lines of submitted code per month)
        - Enhance code security with security vulnerability scanning
          (limit 50 project scans per month)
        - Get answers about your AWS account resources
          (limit 25 queries per month)
        - Diagnose common errors in the console (included)
        - And more

    Q developer Pro Tier: $19/mo.  per user
      - Everything included in the free tier, plus:
      - Manage users and policies with enterprise access controls
      - Customize Amazon Q to your code base to get even better suggestions (preview)
      - High limits of advanced features



  Amazon Q Business
    - Amazon Q Business is a generative AI–powered assistant that can answer questions, provide
      summaries, generate content, and securely complete tasks based on data and information in
      your enterprise systems.
    - It empowers employees to be more creative, data-driven, efficient, prepared, and productive.
    Unified search experience across your knowledge
      - With over 40 built-in secure connectors to popular enterprise applications and document repositories,
        Amazon Q Business unites your disparate data and content silos, offering a unified search experience
        across all your knowledge.
    Automate tasks with Amazon Q Apps
      - users can generate apps in a single step from their conversation with Amazon Q Business or by
        describing their requirements in their own words.
    Share and discover apps within your company
      - allows employees to easily share apps they built and publish them in a central library for others
        to leverage

    Amazon Q Business Lite $3 per user/mo.
      - The Amazon Q Business Lite subscription provides users access to basic functionality such as asking
        questions and receiving permission-aware responses.

     Amazon Q Business Pro $20 per user/mo.
       - The Amazon Q Business Pro subscription provides users access to the full suite of Amazon Q Business capabilities,
         including access to Amazon Q Apps, and Amazon Q in QuickSight (Reader Pro).



-------------------------------------------
https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/what-is.html
What is Amazon Q Developer?

  - Amazon Q Developer is a generative artificial intelligence (AI) powered conversational assistant that can help you understand,
    build, extend, and operate AWS applications.
  - You can ask questions about AWS architecture, your AWS resources, best practices, documentation, support, and more.
  - Amazon Q is constantly updating its capabilities so your questions get the most contextually relevant and actionable answers.

  - When used in an integrated development environment (IDE), Amazon Q provides software development assistance.
  - Amazon Q can chat about code, provide inline code completions, generate net new code, scan your code for security
    vulnerabilities, and make code upgrades and improvements, such as language updates, debugging, and optimizations.

  - Amazon Q is powered by Amazon Bedrock, a fully managed service that makes foundation models (FMs) available through an API.
  - The model that powers Amazon Q has been augmented with high quality AWS content to get you more complete, actionable,
    and referenced answers to accelerate your building on AWS.

-------------------------------------------

 Mastering Amazon Q Business Edition: From Application Creation to AI Assistant - Session 1
   https://www.youtube.com/watch?v=pOqBga80Wfk

  Need a:
     application
     index
     retriever
     data source
     -> after creating, you can ask questions to

     data sources include:
       https://docs.aws.amazon.com/amazonq/latest/qbusiness-ug/connectors-list.html
       S3, Databases (MySQL, Oracle DB, SQL Server, IBM DB2 RDS [MySQL, Oracle, PosgreSQL, MS SQL Server], Auroa),
       Jira, Github, MS Sharepoint Server, Slack, Salesforce Online, ....


  What is Amazon Q business?
    https://docs.aws.amazon.com/amazonq/latest/qbusiness-ug/what-is.html
    - fully managed generative AI powered assistant that you can configure to answer questions,
      generate content, generate summaries, and complete tasks based on data in your enterprise
    - It allows end users to receive immediate, permissions-aware responses from enterprise data sources with citations,
      for use cases such as IT, HR, and benefits help desks.


 Create a policy for the Q app:
 IAM role for an Amazon Q Business
 https://docs.aws.amazon.com/amazonq/latest/qbusiness-ug/create-application-iam-role.html

  - provides cloudwatch and cloudwatch logs access
  - change: region to us-east-1, account_id: <account_id>
  - policy name: twit-qbusiness-policy

{
    "Version": "2012-10-17",
    "Statement": [{
            "Sid": "AmazonQApplicationPutMetricDataPermission",
            "Effect": "Allow",
            "Action": [
                "cloudwatch:PutMetricData"
            ],
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "cloudwatch:namespace": "AWS/QBusiness"
                }
            }
        },
        {
            "Sid": "AmazonQApplicationDescribeLogGroupsPermission",
            "Effect": "Allow",
            "Action": [
                "logs:DescribeLogGroups"
            ],
            "Resource": "*"
        },
        {
            "Sid": "AmazonQApplicationCreateLogGroupPermission",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup"
            ],
            "Resource": [
                "arn:aws:logs:{{region}}:{{account_id}}:log-group:/aws/qbusiness/*"
            ]
        },
        {
            "Sid": "AmazonQApplicationLogStreamPermission",
            "Effect": "Allow",
            "Action": [
                "logs:DescribeLogStreams",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": [
                "arn:aws:logs:{{region}}:{{account_id}}:log-group:/aws/qbusiness/*:log-stream:*"
            ]
        }
    ]
}

# create a IAM role with below custom trust policy
  - change: region to us-east-1, account_id: <account_id>

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AmazonQApplicationPermission",
      "Effect": "Allow",
      "Principal": {
        "Service": "qbusiness.amazonaws.com"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "aws:SourceAccount": "{{account_id}}"
        },
      "ArnLike": {
        "aws:SourceArn":"arn:aws:qbusiness:{{region}}:{{account_id}}:application/*"
      }
      }
    }
  ]
}

# copy Role ARN for use in Application code

AWS Boto3 QBusiness Client docs
  - lists QBusiness Client methods included create_application, create_index, create_retriever

  code:

pip3 install pyyaml
import time
import botocore
import yaml
# simplified means for setting up AWS IAM Identity Center (formally AWS SSO [Single Sign-On]) access
#  https://pypi.org/project/aws-sso-util/
from aws_sso_lib import get_boto3_session


Q_APP_NAME = 'ATwitchDemoApp'


def get_q_application(client, q_app_name):
    print('Check for an existing Q Application ...")
    try:
        response = client.list_applications(
            maxResults=100,
        )
        for app in response['applications']:
            if app['displayName'] == q_app_name:
                return True
        return False
    except: botocore.exceptions.ClientError as error:
        print(error.response(['Error']['Code'])':

# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/qbusiness/client/create_application.html
def create_q_application(client, q_app_desc, q_app_name, q_app_role_arn, q_environment):
    print('Creating a Q Application ...")
    # TBD: add a check to see q_app already exists
    try:
        response = client.create_application(
            displayName=q_app_desc,
            description=q_app_name,
            roleArn=q_app_role_arn,
            tags=[
                { 'key': 'owner', 'value': 'jsmith' },
                { 'key': 'ownerEmail', 'value': 'jsmith@gmail.com' },
                { 'key': 'environment', 'value': q_environment },
            ]

        )
        return response['applicationArn'], response['applicationId']

    except: botocore.exceptions.ClientError as error:
        if error.response(['Error']['Code'] == 'ConflictException'):
            print('A Q Application with that name already exists')
            exit(1)
        else:
           print(error.response['Error']['Code'])

# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/qbusiness/client/create_index.html
def create_q_index(client, q_app_id, q_index_cap_units, q_index_desc, q_index_name, q_environment):
    print('Creating a Q Index ...")
    try:
        response = client.create_index(
            applicationId=q_app_id,
            displayName=q_index_name,
            type='STARTER',
            description=q_index_descr,
            capacityConfiguration={ 'units': q_index_cap_units },
            tags=[
                { 'key': 'owner', 'value': 'jsmith' },
                { 'key': 'ownerEmail', 'value': 'jsmith@gmail.com' },
                { 'key': 'environment', 'value': q_environment },
            ]
        )

        return response['indexArn'], response['indexId']

    except: botocore.exceptions.ClientError as error:
        print(error.response['Error']['Code'])

def create_q_retriever(client, q_app_id, q_index_id, q_retriever_name, q_retriever_type, q_environment):
    print('Creating a Q Retriever ...")
    try:
        response = client.create_retriever(
            applicationId=q_app_id,
            configuration={
                'nativeIndexConfiguration': {
                    'indexId': q_index_id,
                }
            },
            displayName=q_retriever_name,
            type=q_retriever_type,
            tags=[
                { 'key': 'owner', 'value': 'jsmith' },
                { 'key': 'ownerEmail', 'value': 'jsmith@gmail.com' },
                { 'key': 'environment', 'value': q_environment },
            ]
        )

        return response['retrieverArn'], response['retrieverId']

    except: botocore.exceptions.ClientError as error:
        print(error.response['Error']['Code'])
    except: botocore.exceptions.ParamValidationError as error:
        raise ValueError('The parameters you provided are incorrect: {}'.format(error))



def create_q_web_experience(client, q_app_id, q_web_prompt_mode, q_web_subtitle, q_web_title, q_web_welcome_msg, q_environment):
    print('Creating a Q Retriever ...")
    try:
        response = client.create_web_experience(
            applicationId=q_app_id,
            samplePromptsControlMode=q_web_prompt_mode,
            subtitle=q_web_subtitle,
            title=q_web_title,
            welcomeMessage=q_web_welcome_msg,
            tags=[
                { 'key': 'owner', 'value': 'jsmith' },
                { 'key': 'ownerEmail', 'value': 'jsmith@gmail.com' },
                { 'key': 'environment', 'value': q_environment },
            ]
        )

        return response['webExperienceArn'], response['webExperienceId']

    except: botocore.exceptions.ClientError as error:
        print(error.response['Error']['Code'])



def main():
    with open ('config.yaml', 'r') as f:
       config = yaml.safe_load(f)


    sso_start_url = config['config']['sso_start_url']
    sso_region    = config['config']['sso_region']
    account_id    = config['config']['account_id']
    role_name     = config['config']['role_name']
    aws_region    = config['config']['aws_region']
    q_business_role_arn    = config['config']['q_business_role_arn']
    environment   = config['config']['environment']

    # connect to the demo account using aWS Identity Center
    boto3_session = get_boto3_session (sso_start_url, sso_region, account_id, role_name, region=aws_region, login=True)

    # create a boto3 client for QBusiness service
    q_client = boto3_session.client('dbusiness')

    # check if the application already exists
    application_exists = get_q_application(q_client, q_app_name)

    # create a q application
    #  https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/qbusiness/client/create_application.html
    if application_exists is True:
        print ('A Q application already exists. Exiting.')
    else:
        q_app_arn, q_app_id = create_q_application(q_client, 'A Twitch Demo App', Q_APP_NAME, app_role_arn, environment):


    # Create a Q Index
    q_index_arn, q_index_id = create_q_index(q_client, q_app_id, 1, 'A Twitch Demo Index',
                                             'ATwitchDemoIndex', environment):

    # Wait for Index completion before creating a retriever
    #  QBusiness does not current have get_waiter method - using a while loop instead
    index_created = False
    while not index_created:
        try:
            response = q_client.get_index(applicationId=q_app_id, indexId=q_index_id)
            if response['status'] == 'ACTIVE':
                index_created = True
        except: botocore.expections.ClientError as error:
            print(error.response['Error']['Code'])

        print('Waiting on the Q Index to be created ...')
        time.sleep(30)


    # Create a Q retriever
    q_retriever_arn, q_retriever_id = create_q_retriever(client, q_app_id, q_index_id, 'ATwitchDemoRetriever,
                                                         'NATIVE_INDEX', q_environment):

    # Create a Q web experience
    q_web_arn, q_web_id = create_q_web_experience(q_client, q_app_id, 'DISABLED', 'A Twictch Demo Web Experience', 'A Twitch Demo Web Experience', 'Say \'Hi\' to Gordo - he\'s here to help!', environment):

    # web q crawler the application - deployed via boto3
    # in tutorial, data source added in via console

    # close the client
    q_client.close()

main()


-------------------------------------------
-------------------------------------------
-------------------------------------------
