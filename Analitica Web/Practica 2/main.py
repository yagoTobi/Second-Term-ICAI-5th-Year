from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId

import os
import pprint

load_dotenv(find_dotenv())

password = os.environ.get("MONGODB_PWD")

connection_string = (
    f"mongodb+srv://ytobio66:{password}@clusteranalitica.tjuav4q.mongodb.net/"
)
client = MongoClient(connection_string)

dbs = client.list_database_names()
test_db = client.test
collections = test_db.list_collection_names()


def insert_test_doc():
    collection = test_db.test
    test_document = {
        "name": "Yaguete",
        "type": "Test",
    }
    inserted_id = collection.insert_one(test_document).inserted_id


# insert_test_doc()

production = client.production
person_collection = production.person_collection


# * Insert a document
def create_documents():
    first_names = ["Yago", "Luis", "Jesus"]
    surnames = ["Tobio", "Bueno", "Lopez"]
    ages = [23, 23, 22]

    docs = []

    for first_name, surname, age in zip(first_names, surnames, ages):
        doc = {"first_name": first_name, "surname": surname, "age": age}
        docs.append(doc)

    person_collection.insert_many(docs)


# * create_documents()


# ? - How to read documents.
def find_all_people():
    # ? - People is actually a cursor. Which is pretty cool.
    # ? -You can also print list(people) and that's going to give you the array
    people = person_collection.find()

    for person in people:
        print(person)


# find_all_people()


# * Read the document
def find_yago():
    yago = person_collection.find_one({"first_name": "Yago"})
    print(yago)


# ? - find_yago()


def count_all_people():
    count = person_collection.count_documents(filter={})
    # count = person_collection.find().count()
    print(f"Number of people: {count}")


#count_all_people()


# * Find by id
def get_person_by_id(person_id):
    _id = ObjectId(person_id)
    person = person_collection.find_one({"_id": _id})
    print(person)


#get_person_by_id("65c3b1514854cdde930a20d5")


def get_age_range(min_age, max_age):
    # ? - Check out the operators of MongoDb
    query = {
        "$and": [
            # ? - Let's pass the queries:
            {"age": {"$gte": min_age}},  # ? - GTE -> Greater than equal
            {"age": {"$lte": max_age}}
        ]
    }
    people = person_collection.find(query).sort("age")
    for person in people:
        print(person)

get_age_range(18, 24)


def project_columns(): 
    columns = {"_id": 0, "first_name":1, "last_name": 1}
    people = person_collection.find({}, columns)
    for person in people: 
        print(person)

# ! - Update: 
def update_person_by_id(person_id): 

    _id = ObjectId(person_id)

    #all_updates = {
    #    "$set": {"new_field": True}, 
    #    "$inc": {"age": 1}, 
    #    "$rename": {"first_name": "first", "surnames": "last" }
    #}
    #person_collection.update_one({"_id": _id}, all_updates)
    # ? - Para quitar elementos 
    person_collection.update_one({"_id": _id}, {"$unset": {"new_field": ""}})
update_person_by_id("65c3b1514854cdde930a20d5")
find_all_people()