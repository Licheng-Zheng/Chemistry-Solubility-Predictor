import PyPDF2, matplotlib
from mefunctions import detect_if_new, convert_ugliness

# This function will be used when displayed to a viewed because text files can't encode subscripts (bruh)
from mefunctions import get_sub

# creating a pdf file object
pdfFileObj = open('big-copy-of-solubility.pdf', 'rb')

# creating a pdf reader object
pdfReader = PyPDF2.PdfReader(pdfFileObj)

# collects the number of pages that we will have to iterate through
pages = len(pdfReader.pages)

# this is the list with all the information that is to be put into the text file, refreshed with every new element
thingy = []

# Holds the values of the soluble amount and temperature to calculate the line of best fit
current_data = []

# this is the variable that determines if the compound has just been stated (used to find compound names)
immediately_after_compound = True

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

            print(thingy)

            if "continued" in page_number_whatever[line]:
                continue

            thingy = list(map(str, thingy))

            with open('data.txt', 'a', encoding="utf-8") as f:
                f.write("".join(thingy))
                f.write("\n")

            # Sets the data_comes_next variable to false, so we do not put garbage data into our data pile
            data_comes_next = False

            # refreshes the list which holds the information
            # Tells PDF Reader that it's the start of a new Compound
            # Adds on the chemical composition of the compound
            # adds name, indicating that what comes next are the names of the compound
            thingy = ["compound:", get_sub(page_number_whatever[line]), "names:"]

            # Makes current_data variable able to accept new information (for a new compound)
            current_data = []

            # makes the immediately_after_compound boolean True, telling code that what comes next are the names
            immediately_after_compound = True

        # This one finds if we are immediately after the compound, and we are not at the "RN" section
        # (we do not use or want the RN section) Also does not let "Solubility" through
        elif immediately_after_compound and "RN" not in page_number_whatever[line] and "Solu" \
                not in page_number_whatever[line]:

            # At this point, only the names of the compound should be getting through, so this will only append
            # the common names of the compound (in multiple languages too apparently)
            thingy.append(page_number_whatever[line])

            # this is appended so when we make our GUI, we can split the named based on the new (because some of the
            # names have spaces in them so we can't split based on spaces)
            thingy.append("new")

        # Finds if RN is in the current line
        elif "RN" in page_number_whatever[line]:
            # This tells us that the following information is no longer the names of the compound
            immediately_after_compound = False

            # Writes Down "Data" and starts to record the moles dissolved and the temperature (alternating)
            thingy.append("data:")

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
                    temperature = temperature

                try:
                    # converts the soluble amount to a number that machine learning can be used on
                    # (Look at me functions convert_ugliness)
                    new_soluble = convert_ugliness(soluble_amount)

                    # this is saved as important data
                    important_data = [soluble_amount, temperature]

                    # current data is used to calculate curve/line of best fit
                    current_data.append([new_soluble, temperature])

                    # important data is appended, this is the part for the user to see
                    thingy.append(important_data)

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

print(thingy)

thingy = list(map(str, thingy))

with open('data.txt', 'a', encoding="utf-8") as f:
    f.write("".join(thingy))
    f.write("\n")

# closing the pdf file object (save memory!)
pdfFileObj.close()
f.close()
