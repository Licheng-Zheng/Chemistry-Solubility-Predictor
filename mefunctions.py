# No idea what the purpose of this import is, but I'm too scared to remove it 
import periodictable

def detect_if_new(current_line):
    '''
    Takes the current_line, read by the pdf reader, and detects if its a new compound or not. If its a new compound, it begins anew and saves the previous compound's data
    '''

    # this is the line that the code is currently on, it is split into multiple pieces to be tested to see if it's a new compound
    current_line = current_line.split()

    # try statement makes it so that if an error is thrown, the program is not thrown
    try:
        # checks if there is a decimal in the first line (if it is a new compound, there should be)
        if "." in current_line[0]:

            # replaces the decimal with a 0 to allow it to be converted into an integer (if it is a compound)
            first = current_line[0].replace(".", "0")

            # try and except statements try one line of code, and if an error occurs, False is returned
            try:

                # tries making the thing into an integer, if it works, then it passes the test, it won't be an integer if its a new compound, or basically anything to be honest 
                int(first)

            except ValueError:

                # if the thing does not int, it is not a new compound, so false is returned
                return False
        else:

            # False is returned if there isn't a decimal in the first line (it is not a new compound)
            return False

        # If the second line is not a string, they are not elements, so it will not be a new compound
        if str(type(current_line[1])) != "<class 'str'>":
            return False

    # idk why index errors occur sometimes, it's not like there's an empty line or smth, but it does
    # So if theres an index error, it returns false
    except IndexError:
        return False

    # when all tests have been passed, True is returned, indicating that it is a new compound
    return True


# function to convert to subscript
def get_sub(letters):
    '''
    Gets the subscript for the amount of elements in a compound, and changes it to a whole number (not a monkey number), this is added to the list, and makes it machine readable
    '''

    # Receives a list of characters (the compound formula) and makes all of them into a string
    letters = list(map(str, letters.split()))

    letters = letters[1::]

    # The database to convert from a normal number into a subscript
    search = "0123456789"
    subscripted = "₀₁₂₃₄₅₆₇₈₉"

    # For each of the numbers, replaces it with its subscript
    for section in range(len(letters)):

        # Replaces all spaces in all the letters with a no space (space bad >:C)
        letters[section] = letters[section].replace(" ", "")

        # For all the characters in the search string (the numbers), checks if they are present in the section
        # that the code is currently on
        for character in search:
            if character in letters[section]:

                # Finds the character that will be replaced by the subscript
                place = search.index(character)

                # Recreates the section with the numbers swapped out with subscripts
                letters[section] = letters[section].replace(search[place], subscripted[place])

    # Joins everything in the list and returns the entire compound
    return "".join([str(item) for item in letters])


# Takes the scientific notation (1.87E-5) and converts it into a number for modelling purposes
def convert_ugliness(scientific_notation):
    '''
    Changes it from scientific notation to a number, I'm pretty sure I never use this function because I found out it doesn't matter, but yea it is what it is
    '''

    # Splits the thing at the E, the actual value and the exponent for the 10
    part_1, part_2 = list(map(float, scientific_notation.split("E")))

    # Might result in a very small difference from the actual value caused by python being weird at math
    the_number = part_1 * (10 ** part_2)

    return the_number


# Finds the atomic number of each element in a compound and the number and then returns an array
def find_element(letters):
    '''
    Changes it from the letters in the compound and the number of each atom into a list of 118 elements, each element representing the number of that element in the compound'''
    to_return = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    element_table = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                     'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                     'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                     'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
                     'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
                     'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es',
                     'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
                     'Lv', 'Ts']

    copied = letters
    try:
        letters = list(map(str, letters.split()))

        letters = letters[1::]

        letters = str(letters[0])

    except IndexError:
        letters = copied

    # stores all the numbers, so we can iterate through this list
    search = "0123456789"

    # some elements have more than 1 letter, so we need to account for that
    current_letters = []

    # In some compounds, there will be a bunch of a certain element, so we need to store the number of the element
    number = []

    # iterates through all the letters in the compound name
    for letter in letters:

        # if the letter is uppercase, that means it is a new element
        if letter.isupper():
            # finds the element that is current be used and finds its index in the element's table. This is used to
            # update the location of the to_return array
            if len(current_letters) > 0:
                index_number = element_table.index("".join(current_letters))

                # if there is a value in number, that means there is more than 1 of the given element, this is put into
                # the to_return array, if the to_return number is empty, that means there is only one of the element,
                # so one is appended
                if len(number) > 0:
                    to_return[index_number] += int("".join(number))

                else:
                    to_return[index_number] += 1

                # resets everything for the new element
                current_letters = []
                number = []

                current_letters.append(letter)

            else:
                # appends the first uppercase letter of the element
                current_letters.append(letter)

        # if the letter is a number, then the number will be appended, in the case of the else statement, only lower
        # case letters are appended (the first one is an uppercase though)
        elif letter in search:
            number.append(letter)
        else:
            current_letters.append(letter)

    index_number = element_table.index("".join(current_letters))

    # if there is a value in number, that means there is more than 1 of the given element, this is put into
    # the to_return array, if the to_return number is empty, that means there is only one of the element,
    # so one is appended
    if len(number) > 0:
        to_return[index_number] += int("".join(number))
    else:
        to_return[index_number] += 1

    return to_return


# created this thing on the side, just creates a table of all the elements to be used in the find_element function
def create_element_table():
    '''
    I use this for creating the list find_elements (the element_table), so it's basically never used :/'''
    # initiates a list to be appended to
    element_list = []

    for x in range(1, 118):
        # get the element's symbol from periodictable module, and appends it to a list
        element = periodictable.elements[x]
        element_list.append(element.symbol)

    # returns the list so we can use it
    return element_list
