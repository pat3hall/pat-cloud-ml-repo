from json import load, dump, dumps
import copy
from datetime import date, timedelta

json_file_name = "template.json"

# This variable must be defined and populated for tests.
template = None
with open(json_file_name) as json_file:
    template = load(json_file)

def gather_property_info():
    print("Address")
    street_1 = input("Street 1: ")
    street_2 = input("Street 2: ")
    city = input("City: ")
    state = input("State: ")
    zip_code = input("ZIP: ")

    print("\nDetails")
    square_footage = int(input("Square Footage: ").replace(",",""))
    bedrooms = int(input("Bedrooms: "))
    bathrooms = int(input("Bathrooms: "))
    ammenities = list(
        map(str.strip, input("Ammenities (use | between items): ").split("|"))
    )

    return {
        "address": {
            "street_1": street_1,
            "street_2": street_2,
            "city": city,
            "state": state,
            "zip": zip_code,
        },
        "square_footage": square_footage,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "ammenities": ammenities,
    }

def gather_referrer_info():
    has_referrer = input("Was there a referrer? (Y or N): ")
    if has_referrer.lower().startswith("y"):
        return input("Referrer: ")
    else:
        return None

def main():
    # Place all logic in here (or in functions called here)
    output = copy.deepcopy(template)
    property_info = gather_property_info()
    referrer = gather_referrer_info()
    output["listing_date"] = str(date.today() + timedelta(days=1))
    output["property"] = property_info
    output["broker"]["referrer"] = referrer

    output_file_name = property_info["address"]["street_1"].replace(" ", "-") + ".json"
    output_indent_file_name = property_info["address"]["street_1"].replace(" ", "-") + "-indent.json"

    print(output)

    with open(output_file_name, "w") as output_file:
        dump(output, output_file)
        print(f"Created {output_file_name}")

    json_object = dumps(output, indent=4)

    with open(output_indent_file_name, "w") as output_file:
        output_file.write(json_object)
        print(f"Created {output_indent_file_name}")

if __name__ == "__main__":
    main()

