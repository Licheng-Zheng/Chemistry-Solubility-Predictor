# imports code to us
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import PyPDF2
from mefunctions import detect_if_new, convert_ugliness, find_element
import csv

# creating a pdf file object
pdfFileObj = open('big-copy-of-solubility.pdf', 'rb')

# creating a pdf reader object
# pdfReader = PyPDF2.PdfReader(pdfFileObj)

# written by Coralyn (thank you for your contribution), this is now a group project
pdfReader = PyPDF2.PdfReader(pdfFileObj)

# collects the number of pages that we will have to iterate through
pages = len(pdfReader.pages)

# this is the list with all the compound information
compound_elements = []


# this is the list that we append all the data to
compound_elements_information = []

# Holds the values of the soluble amount and temperature to calculate the line of best fit
current_data = []

# this tells us if the data comes next (it is True when it reads Moles in the line, indicating that the next line
# be the start of the data entry, set to false after a new compound is detected
data_comes_next = False


# iterates through all the pages
for page in range(pages):

    # the current page the code is iterating through
    current_page = pdfReader.pages[page]

    # creates a list of all the information on the page
    page_number_whatever = current_page.extract_text().split("\n")

    # finds the number of lines on the current page
    lines_on_current_page = len(page_number_whatever)

    # iterates through all the lines on the current page
    for line in range(lines_on_current_page):

        # Removes all lines with Handbook or solutions in it (it's the page header, this stuff bad)
        if "Handbook" in page_number_whatever[line]:
            continue
        if "Solutions" in page_number_whatever[line]:
            continue

        # this function is used to see whether a new compound has appeared (read the me_functions.py file)
        if detect_if_new(page_number_whatever[line]):

            # if there is a continued in the line, that means it is a continuation of the previous compound, so
            # the data is just added on to the previous data points
            if "continued" in page_number_whatever[line]:
                continue

            # every entry into the compound_elements list is made into a string
            compound_elements = list(map(str, compound_elements))

            # if the data is not ok, "Delete" would have been put into the compound_elements list
            # this will cause the process to be restarted when a new_compound comes up
            if "Delete" in compound_elements:
                compound_elements = []
                compound_elements_information = []

            data_in_a_list = []

            with open('information.csv', 'a', encoding="utf-8", newline="") as file:

                # This for loop creates the list that will be appended as information into the data file
                for item in compound_elements_information:
                    try:
                        writer = csv.writer(file)
                        item = [float(item[1]), float(item[0])]
                        to_write_to_csv = compound_elements + item

                        if len(to_write_to_csv) == 121:
                            writer.writerow(to_write_to_csv)
                        else:
                            print(to_write_to_csv)
                            pass
                    # print(item)
                    except ValueError:
                        pass

            # the data that needs to be written
            compound_elements_information = []

            # Sets the data_comes_next variable to false, so we do not put garbage data into our data pile
            data_comes_next = False

            # refreshes the list which holds the information
            # Tells PDF Reader that it's the start of a new Compound
            # Adds on the chemical composition of the compound
            # adds name, indicating that what comes next are the names of the compound
            try:
                compound_elements = find_element(page_number_whatever[line])
            except ValueError:
                compound_elements.append("Delete")

            # Makes current_data variable able to accept new information (for a new compound)
            current_data = []

            # makes the immediately_after_compound boolean True, telling code that what comes next are the names
            immediately_after_compound = True

        # When data_comes_next is true, the values that we need come after
        if data_comes_next:

            # We take the current line and split it, creating a list
            data_line = page_number_whatever[line].split()
            if "ns" in data_line:
                continue
            try:
                # from this list, we only take the information we need, the first item is the solubility in mol
                soluble_amount = data_line[0]

                # this value is the temperature in degrees Celsius
                temperature = data_line[2]

                # sometimes, we get "ns" (not stated) as the temperature, so we have a try and except statement
                # we try to turn into an integer (and then into degrees kelvin), if this doesn't work because
                # temperature is not a value that can be turned into an integer ("ns") it just leaves it as is
                try:
                    temperature = int(temperature) + 273
                except ValueError:
                    # nothing happens if the thing cannot be turned into an integer
                    pass

                try:
                    # converts the soluble amount to a number that machine learning can be used on
                    # (Look at me functions convert_ugliness)
                    new_soluble = convert_ugliness(soluble_amount)

                    # this is saved as important data
                    important_data = [soluble_amount, temperature]

                    # current data is used to calculate curve/line of best fit
                    current_data.append([new_soluble, temperature])

                    # important data is appended, this is the part for the user to see
                    compound_elements_information.append(important_data)

                # Just doesn't do anything when something doesn't work (the less than symbol in particular)
                # In that particular case, it just doesn't add it to the data (which is what we want!) (i think)
                except ValueError:
                    pass

            # this prevents code from blowing up when it tries to read a black line
            except IndexError:
                pass

        # before the data comes, "moles" is stated in the line
        if "Moles" in page_number_whatever[line]:

            # Makes data_comes_next true (read if statement above for what this does)
            data_comes_next = True

compound_elements = list(map(str, compound_elements))

data_in_a_list = []

with open('information.csv', 'a', encoding="utf-8", newline="") as file:
    # This for loop creates the list that will be appended as information into the data file
    for item in compound_elements_information:
        try:
            writer = csv.writer(file)
            item = [float(item[1]), float(item[0])]
            to_write_to_csv = compound_elements + item

            if len(to_write_to_csv) == 121:
                writer.writerow(to_write_to_csv)
            else:
                print(to_write_to_csv)
                pass

        except ValueError:
            pass

# closing the pdf file object (save memory!)
pdfFileObj.close()

file.close()
