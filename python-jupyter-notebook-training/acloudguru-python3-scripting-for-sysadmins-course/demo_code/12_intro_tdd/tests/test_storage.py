import tempfile
import pytest

from pgbackup import storage

# create infile fixture that can be passed to multiple tests
@pytest.fixture
def infile():
    # create a temporary file with mode='w+b' (write and byte type) (default TemporaryFile() mode)
    f = tempfile.TemporaryFile()
    # write byte data to temporary file
    f.write(b'Testing')
    # go back beginning of file so read starts at beginning
    f.seek(0)
    return f


def test_storing_file_locally(infile):
    """
    Writes content from one file-like to another
    """
    # create temporary Named out file that is not deleted when closed so it can be re-opened to check contents
    outfile = tempfile.NamedTemporaryFile(delete=False)
    #
    storage.local(infile, outfile)
    # verify outfile contents match infile contents (byte read comparison)
    with open(outfile.name, 'rb') as f:
        assert f.read() == b'Testing'

def test_storing_file_on_s3(mocker,infile):
    """
    Writes content from one file-like to S3
    """
    # a mock client
    client = mocker.Mock()

    storage.s3(client, infile, "bucket", "file-name")

    client.upload_fileobj.assert_called_with(infile, "bucket", "file-name")


