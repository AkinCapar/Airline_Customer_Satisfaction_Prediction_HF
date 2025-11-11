import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)


            result = "Satisfied 😊" if predictions[0] == 1 else "Neutral or Dissatisfied 😞"

            return f"Predicted Customer Satisfaction: {result}"

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 gender: str,
                 customer_type: str,
                 age: int,
                 type_of_travel: str,
                 flight_class: str,
                 flight_distance: int,
                 inflight_wifi_service: int,
                 departure_arrival_time_convinient: int,
                 ease_of_online_booking: int,
                 gate_location: int,
                 food_drink: int,
                 online_boarding: int,
                 seat_comfort: int,
                 inflight_entertainment: int,
                 onboard_service: int,
                 leg_room_service: int,
                 baggage_handling: int,
                 checkin_service: int,
                 inflight_service: int,
                 cleanliness: int,
                 departure_delay: int,
                 arrival_delay: float):
        
        self.gender = gender
        self.customer_type = customer_type
        self.age = age
        self.type_of_travel = type_of_travel
        self.flight_class = flight_class
        self.flight_distance = flight_distance
        self.inflight_wifi_service = int(inflight_wifi_service)
        self.departure_arrival_time_convinient = int(departure_arrival_time_convinient)
        self.ease_of_online_booking = int(ease_of_online_booking)
        self.gate_location = int(gate_location)
        self.food_drink = int(food_drink)
        self.online_boarding = int(online_boarding)
        self.seat_comfort = int(seat_comfort)
        self.inflight_entertainment = int(inflight_entertainment)
        self.onboard_service = int(onboard_service)
        self.leg_room_service = int(leg_room_service)
        self.baggage_handling = int(baggage_handling)
        self.checkin_service = int(checkin_service)
        self.inflight_service = int(inflight_service)
        self.cleanliness = int(cleanliness)
        self.departure_delay = departure_delay
        self.arrival_delay = float(arrival_delay)


        if age > 100:
            age = 99
        
        if age < 0:
            age = 1
        bins = [0, 18, 30, 45, 60, 100] 
        labels = ['0–18', '19–30', '31–45', '46–60', '60+']
        self.age_group = pd.cut([age], bins=bins, labels=labels)[0]
        

        if flight_distance >= 5000:
            flight_distance = 4999
        bins = [0, 1000, 3000, 5000]
        labels = ['Short', 'Medium', 'Long']
        self.flight_distance_group = pd.cut([flight_distance], bins=bins, labels=labels)[0]

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender" : [self.gender],
                "Customer Type": [self.customer_type],
                "Age": [self.age],
                "Type of Travel": [self.type_of_travel],
                "Class": [self.flight_class],
                "Flight Distance": [self.flight_distance],
                "Inflight wifi service": [self.inflight_wifi_service],
                "Departure/Arrival time convenient": [self.departure_arrival_time_convinient],
                "Ease of Online booking": [self.ease_of_online_booking],
                "Gate location": [self.gate_location],
                "Food and drink": [self.food_drink],
                "Online boarding": [self.online_boarding],
                "Seat comfort": [self.seat_comfort],
                "Inflight entertainment": [self.inflight_entertainment],
                "On-board service": [self.onboard_service],
                "Leg room service": [self.leg_room_service],
                "Baggage handling": [self.baggage_handling],
                "Checkin service": [self.checkin_service],
                "Inflight service": [self.inflight_service],
                "Cleanliness": [self.cleanliness],
                "Departure Delay in Minutes": [self.departure_delay],
                "Arrival Delay in Minutes": [self.arrival_delay],
                "AgeGroup": [self.age_group],
                "FlightDistanceGroup": [self.flight_distance_group]
                }
            
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
