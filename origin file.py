import PyPDF2

# creating a pdf file object
pdfFileObj = open('Samuel_H._Yalkowsky_-_Handbook_of_Aqueous_Solubility_Data-CRC_Press_2003[1]-16-1286.pdf', 'rb')

# creating a pdf reader object
pdfReader = PyPDF2.PdfReader(pdfFileObj)

# collects the number of pages that we will have to iterate through
pages = len(pdfReader.pages)


# iterates through all the pages
for page in range(pages):

    # the current page the code is iterating through
    current_page = pdfReader.pages[page]

    # creates a list of all the information on the page
    page_number_whatever = current_page.extract_text().split("\n")

    print(page_number_whatever)


# closing the pdf file object (save memory!)
pdfFileObj.close()
