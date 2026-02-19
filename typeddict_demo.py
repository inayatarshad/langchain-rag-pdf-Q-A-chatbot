from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person = Person(name="Inayat", age=21)

print("new_person:", new_person)