marks = [78, 85, 92, 67, 88]

for m in marks:
    if m >= 80:
        print(m,"-good")
    else:
        print(m,"-improve pannunga")
        
def average(nums):
        return sum(nums)/len(nums)

print("average:", average(marks))

student = {"name": "arun","marks": marks}
print(student["name"], average(student["marks"]))