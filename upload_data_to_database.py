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

def update_csv_from_firebase():
    firebase_data = ref.get()
    with open('info.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        csv_data = {row['id']: row for row in csv_reader}
        
        for key, value in firebase_data.items():
            if key in csv_data:
                csv_data[key].update(value)
            else:
                csv_data[key] = value
                
    with open('info.csv', mode='w', newline='') as csv_file:
        fieldnames = ['id', 'name', 'major', 'total_attendance']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_data.values():
            writer.writerow(row)

#### IMPORTANT #####
FIREBASE_TO_CSV = True # Change to False if there is new student
if FIREBASE_TO_CSV:
    update_csv_from_firebase()
else:
    with open('info.csv', mode='r') as csv_file:
        data = {}

        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            row['total_attendance'] = int(row['total_attendance'])
            data[row['id']] = row


    for key, value in data.items():
        ref.child(key).set(value)