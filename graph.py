import re
import matplotlib.pyplot as plt
k=5
filepath = str(k)+"-shotlog.txt"
text ="" 
with open(filepath, "r") as f:
    text = f.read()
matches = re.findall(r"Dev loss average: tensor\(([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)),", text)
print(len(matches))

losses = [float(x[0]) for x in matches]

plt.plot(losses, label="Avg Dev Loss")

matches =  re.findall(r"Training loss average: tensor\(([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)),", text)
print(len(matches))

losses = [float(x[0]) for x in matches]

plt.plot(losses, label="Avg Training Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel=("Loss")
plt.savefig(str(k)+"-shotLoss.png")
plt.close()


