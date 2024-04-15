import csv
import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("secret/serviceAccountKey.json")
firebase_admin.initialize_app(
	cred,
	{
		"databaseURL": "https://face-identification-real-time-default-rtdb.firebaseio.com/"
	}
)

ref = db.reference("Students")

# Open the CSV file
with open('info.csv', mode='r') as csv_file:
    data = {}

    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        row['starting_year'] = int(row['starting_year'])
        row['total_attendance'] = int(row['total_attendance'])
        row['year'] = int(row['year'])
        data[row['id']] = row


for key, value in data.items():
	ref.child(key).set(value)