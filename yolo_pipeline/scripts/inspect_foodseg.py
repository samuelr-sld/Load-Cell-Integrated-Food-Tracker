from datasets import load_dataset

dataset = load_dataset("EduardoPacheco/FoodSeg103")

print(dataset)
print(dataset["train"][0])