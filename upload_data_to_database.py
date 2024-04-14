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

data = {
	"321654":
		{
			"name": "Phan Tai Duc",
			"major": "AI",
			"starting_year": 2023,
			"total_attendance": 6,
			"standing": "G",
			"year": 4,
			"last_attendance_time": "2024-04-14 12:54:34"
		},
	"852741":
		{
			"name": "Emily Blunt",
			"major": "Economics",
			"starting_year": 2022,
			"total_attendance": 5,
			"standing": "G",
			"year": 3,
			"last_attendance_time": "2024-04-14 12:54:34"
		},
	"963852":
		{
			"name": "Elon Musk",
			"major": "Physics",
			"starting_year": 2023,
			"total_attendance": 6,
			"standing": "G",
			"year": 4,
			"last_attendance_time": "2024-04-14 12:54:34"
		},
}

for key, value in data.items():
	ref.child(key).set(value)