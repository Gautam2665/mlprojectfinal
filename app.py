from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from datetime import datetime, date, timedelta
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError

app = Flask(__name__)


# Load trained model
base_model = joblib.load("base_model.pkl")
holiday_model = joblib.load("holiday_model.pkl")

# Google Calendar API Setup
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
HOLIDAY_CALENDAR_ID = "en.indian#holiday@group.v.calendar.google.com"

# Load dataset
df = pd.read_csv("Clean_flight_data.csv")
df["days_left"] = df["days_left"].astype(int)
df["class"] = df["class"].str.capitalize()

def get_holidays():
    """Fetch upcoming public holidays from Google Calendar."""
    creds = None
    token_file = "token.json"

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)

        with open(token_file, "w") as token:
            token.write(creds.to_json())

    service = build("calendar", "v3", credentials=creds)
    now = datetime.now().isoformat() + "Z"
    end_date = (datetime.now() + timedelta(days=365)).isoformat() + "Z"

    events_result = service.events().list(
        calendarId=HOLIDAY_CALENDAR_ID,
        timeMin=now,
        timeMax=end_date,
        maxResults=50,
        singleEvents=True,
        orderBy="startTime",
    ).execute()

    return {event["start"]["date"]: event["summary"] for event in events_result.get("items", [])}

holidays = get_holidays()  # Load holidays once at startup

def enrich_features(df, user_date=None, holidays_map=None):
    df = df.copy()
    if user_date:
        # Your form sends: YYYY-MM-DD
        travel_date_obj = datetime.strptime(user_date, "%Y-%m-%d").date()
        day_of_week = travel_date_obj.weekday()
        is_holiday = 1 if travel_date_obj.strftime("%Y-%m-%d") in holidays_map else 0
        df["day_of_week"] = day_of_week
        df["is_holiday"] = is_holiday
    else:
        df["day_of_week"] = 0
        df["is_holiday"] = 0
    return df

AIRLINE_LOGOS = {
    "Air India": "/static/logos/air-india.png",
    "Indigo": "/static/logos/indigo.png",
    "SpiceJet": "/static/logos/spicejet.png",
    "Vistara": "/static/logos/vistara.png",
    "GO FIRST": "/static/logos/goair.png",
    "AirAsia": "/static/logos/airasia.png",
}

airport_lookup = {
    "Mumbai": {"code": "BOM", "name": "Chhatrapati Shivaji Maharaj International Airport"},
    "Delhi": {"code": "DEL", "name": "Indira Gandhi International Airport"},
    "Chennai": {"code": "MAA", "name": "Chennai International Airport"},
    "Bangalore": {"code": "BLR", "name": "Kempegowda International Airport"},
    "Hyderabad": {"code": "HYD", "name": "Rajiv Gandhi International Airport"},
}


def recommend_flights(source, destination, flight_class, travel_date, holidays,sort_by="cheap", top_n=20):
    """Recommend flights with ML-based holiday price adjustment based on user input."""

    today_date = date.today()

    try:
        travel_date_obj = datetime.strptime(travel_date, "%Y-%m-%d").date()
    except ValueError:
        print("Invalid date format received:", travel_date)
        return []  # Invalid date

    days_left = (travel_date_obj - today_date).days
    print(f"Filtering flights for: {source} -> {destination}, Class: {flight_class}, Travel Date: {travel_date} ({days_left} days left) , Filter:{sort_by}")

    df["days_left"] = df["days_left"].astype(int)

    # Filter available flights based on user input
    filtered_flights = df[
        (df["source_city"].str.lower() == source.lower()) &
        (df["destination_city"].str.lower() == destination.lower()) &
        (df["class"].str.lower() == flight_class.lower()) &
        (df["days_left"] == days_left)
    ].copy()

    if filtered_flights.empty:
        print("No matching flights found.")
        return []

    # Enrich the filtered flights with day_of_week and holiday info
    filtered_flights = enrich_features(filtered_flights, user_date=travel_date, holidays_map=holidays)

    features = [
        "source_city", "destination_city", "airline", "departure_time",
        "arrival_time", "stops", "class", "days_left", "day_of_week", "is_holiday"
    ]

    # Predict the base price
    filtered_flights["predicted_price"] = base_model.predict(filtered_flights[features])

    # Apply holiday adjustment if the flight falls on a holiday
    if filtered_flights["is_holiday"].iloc[0] == 1:
        holiday_price_adjustment = holiday_model.predict(filtered_flights[features])
        filtered_flights["predicted_price"] = (
            filtered_flights["predicted_price"] +
            (holiday_price_adjustment - filtered_flights["predicted_price"]) * 0.75
        )
        
    filtered_flights["airline_logo"] = filtered_flights["airline"].map(AIRLINE_LOGOS)
    filtered_flights["source_code"] = filtered_flights["source_city"].map(lambda x: airport_lookup.get(x, {}).get("code", ""))
    filtered_flights["destination_code"] = filtered_flights["destination_city"].map(lambda x: airport_lookup.get(x, {}).get("code", ""))
    filtered_flights["source_airport"] = filtered_flights["source_city"].map(lambda x: airport_lookup.get(x, {}).get("name", ""))
    filtered_flights["destination_airport"] = filtered_flights["destination_city"].map(lambda x: airport_lookup.get(x, {}).get("name", ""))

    # Sort AFTER assigning all extra display info
    if sort_by == "best":
        sorted_flights = filtered_flights.sort_values(
            by=["stops", "departure_time", "predicted_price"]
        )
    else:  # Default: sort by cheapest
        sorted_flights = filtered_flights.sort_values(
            by=["predicted_price", "days_left"]
        )

    print(f"Flights found: {len(filtered_flights)}")
    return sorted_flights.head(top_n).to_dict(orient="records")


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/flight-details", methods=["GET"])
def flight_details():
    source = request.args.get("source")
    destination = request.args.get("destination")
    flight_class = request.args.get("class")
    travel_date = request.args.get("date")  # Expecting YYYY-MM-DD
    filter_type = request.args.get("sort_by", "cheap")
    if not filter_type:  # If empty, reset to cheap
        filter_type = "cheap" 
    flights = recommend_flights(source, destination, flight_class, travel_date,holidays,sort_by=filter_type)

    print("Flights being sent to template:", flights)  # Debugging step

    return render_template("flight-details.html", flights=flights,source=source,destination=destination,travel_date=travel_date,flight_class=flight_class,sort_by=filter_type)

if __name__ == "__main__":
    app.run(debug=True)