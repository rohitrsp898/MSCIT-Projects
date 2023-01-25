import aiml
import time

time.clock = time.time
kernel = aiml.Kernel()
kernel.learn(r"D:\code\PyCharm\Practicals\Applied Artificial Intelligence\p_1_std-startup.xml")
kernel.respond("LOAD")
# Press CTRL-C to break this loop
while True:
    print(kernel.respond(input("Enter your message >> ")))


# import aiml
# import time
#
# time.clock = time.time
# # Create the kernel and learn AIML files
# kernel = aiml.Kernel()
# kernel.learn("std-startup.xml")
# kernel.respond("load aiml b")
#
# # Press CTRL-C to break this loop
# while True:
#     print(kernel.respond(input("Enter your message >> ")))