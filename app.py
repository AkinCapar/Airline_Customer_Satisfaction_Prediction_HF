import gradio as gr
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


def airline_customer_satisfaction(
        gender,
        customer_type,
        age,
        type_of_travel,
        flight_class,
        flight_distance,
        inflight_wifi_service,
        departure_arrival_time_convinient,
        ease_of_online_booking,
        gate_location,
        food_drink,
        online_boarding,
        seat_comfort,
        inflight_entertainment,
        onboard_service,
        leg_room_service,
        baggage_handling,
        checkin_service,
        inflight_service,
        cleanliness,
        departure_delay,
        arrival_delay
):
    try:
        data = CustomData(
            gender = gender,
            customer_type = customer_type,
            age = age,
            type_of_travel = type_of_travel,
            flight_class = flight_class,
            flight_distance = flight_distance,
            inflight_wifi_service = inflight_wifi_service,
            departure_arrival_time_convinient = departure_arrival_time_convinient,
            ease_of_online_booking = ease_of_online_booking,
            gate_location = gate_location,
            food_drink = food_drink,
            online_boarding = online_boarding,
            seat_comfort = seat_comfort,
            inflight_entertainment = inflight_entertainment,
            onboard_service = onboard_service,
            leg_room_service = leg_room_service,
            baggage_handling = baggage_handling,
            checkin_service = checkin_service,
            inflight_service = inflight_service,
            cleanliness = cleanliness,
            departure_delay = departure_delay,
            arrival_delay = arrival_delay
        )

        df = data.get_data_as_data_frame()

        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)

        return prediction
    
    except Exception as e:
        return f"Error: {str(e)}"


# Gradio UI
demo = gr.Interface(
    fn=airline_customer_satisfaction,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown(["Loyal Customer", "disloyal Customer"], label="Customer Type"),
        gr.Number(label="Age"),
        gr.Dropdown(["Business travel", "Personal Travel"], label="Type of Travel"),
        gr.Dropdown(["Business", "Eco", "Eco Plus"], label="Class"),
        gr.Number(label="Flight Distance"),
        gr.Dropdown(choices=list(range(0, 6)), label="Inflight WiFi Service"),
        gr.Dropdown(choices=list(range(0, 6)), label="Departure/Arrive Time Convenience"),
        gr.Dropdown(choices=list(range(0, 6)), label="Ease of Online Booking"),
        gr.Dropdown(choices=list(range(0, 6)), label="Gate Location"),
        gr.Dropdown(choices=list(range(0, 6)), label="Food and Drink"),
        gr.Dropdown(choices=list(range(0, 6)), label="Online Boarding"),
        gr.Dropdown(choices=list(range(0, 6)), label="Seat Comfort"),
        gr.Dropdown(choices=list(range(0, 6)), label="Inflight Entertainment"),
        gr.Dropdown(choices=list(range(0, 6)), label="On-Board Service"),
        gr.Dropdown(choices=list(range(0, 6)), label="Leg Room"),
        gr.Dropdown(choices=list(range(0, 6)), label="Baggage Handling"),
        gr.Dropdown(choices=list(range(0, 6)), label="Check In Service"),
        gr.Dropdown(choices=list(range(0, 6)), label="Inflight Service"),
        gr.Dropdown(choices=list(range(0, 6)), label="Cleanliness"),
        gr.Number(label="Departure Delay in Minutes"),
        gr.Number(label="Arrival Delay in Minutes"),
        
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Airline Customer Satisfaction Predictor",
    description="This model predicts airline customer is satisfied or not.",
    allow_flagging="never"
)

if __name__ == "__main__":
    
    demo.launch()
