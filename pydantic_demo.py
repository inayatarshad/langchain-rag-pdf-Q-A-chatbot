from pydantic import BaseModel, EmailStr, Field #field cgpa kay liye
class Student(BaseModel):
    name: str
    email: EmailStr  # Use EmailStr for email validation
    cgpa: float = Field(gt=0.0, lt=10.0)

new_student = {'name': 'inayat', 'email': 'inayat@example.com', 'cgpa': 8.5}#create a dictionary
student = Student(**new_student)#** unwrap the same dict and pass its values to the student
#print(student)

student_dict = dict(student)
print(student_dict)
print(student_dict['name'])

student_json = student.model_dump_json()
print(student_json)
