import pytest

# Note: 'cli' module and creat_parser() function  need to be created
from pgbackup import cli

# expected command syntax:
#  pgbackup postgres://bob@example.com:5432/db_one --driver s3 backupsBucket

# url is just the database string
url = "postgres://bob:password@example.com:5432/db_one"

# create 'parser' fixture to replace 'parser = cli.create_parser()' call in each test function
#   and instead pass 'parser' to test function call/def
@pytest.fixture
def parser():
    return cli.create_parser()


def test_parser_without_driver(parser):
    """
    Without a specified driver (e.g. --driver [s3 <bucket>| local <dumpFile>), the parser will exit
    """
    with pytest.raises(SystemExit):
        parser.parse_args([url])

def test_parser_with_driver(parser):
    """
    The parser will exit if it receives a driver Without destination
    """
    with pytest.raises(SystemExit):
        parser.parse_args([url, "--driver", "local"])

def test_parser_with_unknown_driver(parser):
    """
    The parser will exist if the driver name is unknown
    """
    with pytest.raises(SystemExit):
        # should fail becauser 'azure' is not supported
        parser.parse_args([url, "--driver", 'azure', 'destination'])

def test_parser_with_known_driver(parser):
    """
    The parser will not exist if the driver name is known
    """
    for driver in ['local', 's3']:
        assert parser.parse_args([url, '--driver', driver, 'destination'])


def test_parser_with_driver_and_destination(parser):
    """
    The parser will not exit if it receives a driver and destination
    """
    args = parser.parse_args([url, '--driver', 'local', '/some/path'])
    assert args.url == url
    assert args.driver == 'local'
    assert args.destination == '/some/path'

