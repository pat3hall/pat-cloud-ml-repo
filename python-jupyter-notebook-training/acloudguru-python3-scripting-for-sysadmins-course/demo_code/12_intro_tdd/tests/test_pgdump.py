import pytest
import subprocess

from pgbackup import pgdump

url = "postgres://bob:password@example.com:5432/db_one"

def test_dump_calls_pg_dump(mocker):
    """
    Utilize pg_dump with the database URL
    """
    # using Popen class and stub out it's constructor

    # use mocker to stub out 'subprocess.Popen'
    mocker.patch('subprocess.Popen')
    assert pgdump.dump(url)
    # call the mock version Popen passing the expect args list to it
    subprocess.Popen.assert_called_with(['pg_dump', url], stdout=subprocess.PIPE)


def test_dump_handles_oserror(mocker):
    """
    pgdump.dump returns a reasonable error if pg_dump is not installed
    """
    # use mocker to stub out 'subprocess.Popen'
    # if pg_dump is called, but it is not installed, give SystemExit error
    mocker.patch('subprocess.Popen', side_effect=OSError('no such file'))
    with pytest.raises(SystemExit):
        pgdump.dump(url)


def test_dump_file_name_without_timestamp():
    """
    pgdump.dump_file_name returns the name of the database
    """
    # 'dp_one.sql' filename is based on url database name is 'db_one'
    assert pgdump.dump_file_name(url) == 'db_one.sql'


def test_dump_file_name_with_timestamp():
    """
    pgdump.dump_file_name returns the name of the database with timestamp
    """
    timestamp = "2023-09-08T14:47:10"
    # 'dp_one.sql' filename is based on url database name is 'db_one'
    assert pgdump.dump_file_name(url,timestamp) == f"db_one-{timestamp}.sql"

