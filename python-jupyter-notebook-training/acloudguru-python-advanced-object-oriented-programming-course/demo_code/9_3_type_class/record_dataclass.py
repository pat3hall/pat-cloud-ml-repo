import datetime
from datetime import date
from dataclasses import dataclass

@dataclass
class Record():
    artist: str
    title: str
    release_date: datetime.date
    album_color: str = "blue"

record1 = Record("The Beatles", "With the Beatles", f"{datetime.date(1963, 11, 22)}", "red")
print (f"\nrecord1:  {record1}")
record2 = Record("The Beatles", "With the Beatles", f"{datetime.date(1963, 11, 22)}", "red")
print (f"record2:  {record2}")
record3 = Record("The Beatles", "Help", f"{datetime.date(1963, 11, 22)}")
print (f"record3:  {record3}")

print (f"\nrecord1.artist:       {record1.artist}")
assert record1.artist == "The Beatles"
print (f"record1.title:        {record1.title}")
assert record1.title == "With the Beatles"
print (f"record1.album_color:  {record1.album_color}")
assert record1.album_color == "red"
print (f"record1.release_date: {record1.release_date}")
assert record1.release_date == f"{datetime.date(1963, 11, 22)}"

# was not able to get below assert to work because it set "release_date: date: datetime.date.today()"
#   instead of  "release_date: datetime.date: datetime.date.today()"
#assert record3.release_date == datetime.date.today()

print (f"\nrecord3.album_color:  {record3.album_color}")
assert record3.album_color == "blue"
assert record1 == record2
assert record1 != record3



