from matplotlib import pyplot as plt
import seaborn as sns


def reverse_komplement(DNA_streng):
    reverse2 = ""

    #går gjennom hvert element i DNA
    for base in DNA_streng:
        if base == "A":
            reverse2 = "T" + reverse2 # Legger til basen, også resten av lista.
        elif base == "G":
            reverse2 = "C" + reverse2
        elif base == "C":
            reverse2 = "G" + reverse2
        elif base == "T":
            reverse2 = "A" + reverse2
    return reverse2


def palindrome_test(i, virus):

    count = 0
    for posisjon in range(len(virus) - i):
        del_tekst = virus[posisjon:posisjon + i]

        reverse = reverse_komplement(del_tekst)

        if del_tekst == reverse:
            count += 1

    return count


with open("message.txt", "r") as file:
    Measles_Virus = file.read().replace("\n", "")


values = []
for j in range(2, 30):
    values.append(palindrome_test(j, Measles_Virus))

plt.plot(range(2, len(values) + 2), values)
plt.show()
