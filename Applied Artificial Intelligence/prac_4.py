# Probability Join
pofB = float(input("Enter number of C programmers in percentage : "))
pofAandB = float(input("Enter number of C and Java programmers in percentage : "))
pofB = pofB / 100
pofAandB = pofAandB / 100
print("Event A that student is Java Programmer=?")
print("Event B that student is C Programmer=", pofB)
print("Event A and B that is student knowing both C and Java is =", pofAandB)
print("Lets Calculate P(A|B)= P(A and B) / P(B)")
pAgivenB = pofAandB / pofB
print("P(A|B)=", pAgivenB)
print("There are ", pAgivenB * 100, " % chances that the student that knows C also knows Java")
